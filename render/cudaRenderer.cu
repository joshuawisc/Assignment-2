#include <algorithm>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include "circleBoxTest.cu_inl"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define SCAN_BLOCK_DIM  1024  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif



////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float *position;
    float *velocity;
    float *color;
    float *radius;

    int imageWidth;
    int imageHeight;
    float *imageData;
};

// TODO (optimization): delete prefixes and update hitCircles inplace instead
// TODO (optimization): make hitCircles boolean to save space?
int* hitCircles = NULL; // numBlocks arrays of size numCircles
int* prefixes = NULL; // numBlocks arrays of size numCircles
int* hitCirclesList = NULL; // numBlocks arrays of size circles in each block

int blockSize = 64;
int numBlocks = 0;

// Global variable that is in scope, but read-only, for all CUDA
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (We didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "lookupColor.cu_inl"
#include "noiseCuda.cu_inl"

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;
    float *radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // Travel scaled unit length
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}

// kernelAdvanceBouncingBalls
//
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3 + 1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3 + 1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon && oldPosition < 0.0f &&
        fabsf(position[index3 + 1] - oldPosition) < epsilon) { // stop ball
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float *positionPtr = &cuConstRendererParams.position[index3];
    float *velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3 *)positionPtr);
    float3 velocity = *((float3 *)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ((position.y + radius < 0.f) || (position.x + radius) < -0.f || (position.x - radius) > 1.f) {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3 *)positionPtr) = position;
    *((float3 *)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void shadePixel(float2 pixelCenter, float3 p, float4 *imagePtr, int circleIndex) {
    // printf("HERE1\n");
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    /**
    // DELETE //
    float4 newColor;
    newColor.x = 255;
    newColor.y = 0;
    newColor.z = 0;
    newColor.w = 1;


    // Global memory write
    *imagePtr = newColor;
    return;
    // DELETE //
    **/

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES ||
        cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * index;

    // Read position and radius
    float3 p = *(float3 *)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[index];

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // For all pixels in the bounding box
    for (int pixelY = screenMinY; pixelY < screenMaxY; pixelY++) {
        float4 *imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX = screenMinX; pixelX < screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(pixelCenterNorm, p, imgPtr, index);
            imgPtr++;
        }
    }
}

// kernelRenderPixels -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderPixels() {
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth; 
    int imageHeight = cuConstRendererParams.imageHeight; 

    if ((indexX >= imageWidth) || (indexY >= imageHeight))
        return;

    // Get pixelY, pixelX from index
    int pixelY = indexY;
    int pixelX = indexX;

    // For one pixel
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float4 *imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));

    
    //TODO: 
    // Get list
    // Iterate over list of circles 
    for (int i = 0 ; i < cuConstRendererParams.numberOfCircles ; i++) {
        int i3 = 3 * i;

        // Read position and radius
        float3 p = *(float3 *)(&cuConstRendererParams.position[i3]);

        /**
        // Compute the bounding box of the circle. The bound is in integer
        // screen coordinates, so it's clamped to the edges of the screen.
        short imageWidth = cuConstRendererParams.imageWidth;
        short imageHeight = cuConstRendererParams.imageHeight;
        short minX = static_cast<short>(imageWidth * (p.x - rad));
        short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
        short minY = static_cast<short>(imageHeight * (p.y - rad));
        short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

        // A bunch of clamps.  Is there a CUDA built-in for this?
        short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
        short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
        short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
        short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

        float invWidth = 1.f / imageWidth;
        float invHeight = 1.f / imageHeight;
        **/
        shadePixel(pixelCenterNorm, p, imgPtr, i);

    }

}

__global__ void kernelRenderPixelsNew(int* hitCirclesList, int blockSize) {
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth; 
    int imageHeight = cuConstRendererParams.imageHeight; 

    if ((indexX >= imageWidth) || (indexY >= imageHeight))
        return;

    // Get pixelY, pixelX from index
    int pixelY = indexY;
    int pixelX = indexX;

    // For one pixel
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float4 *imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));

    //TODO: 

    // Get block number
    int gridXSize = (imageWidth + blockSize - 1) / blockSize;
    //blockId = rowNumber*width + columnNumber
    int blockId = ((pixelY/blockSize)*gridXSize) + (pixelX/blockSize);

    int calcBlockId = blockIdx.y*gridDim.x + blockIdx.x;
    calcBlockId = blockId;

    // Get list
    // Iterate over list of circles 
    int numCircles = cuConstRendererParams.numberOfCircles;
    int circleId; //

    /**
    if (pixelY == 539 && pixelX == 641)
        printf("BlockID: %d\n", calcBlockId);


    
    if (pixelY == 550 && pixelX == 528) {
        printf("blockId: %d\n", blockId);
        printf("calcBlockId: %d\n", calcBlockId);
        printf("indexList: ");
       
        for (int j = 0 ; j < numCircles ; j++) {
            printf("%d,", hitCirclesList[calcBlockId*numCircles+j]);
            if (hitCirclesList[calcBlockId*numCircles+j] == -1)
                break;
        }
        
        printf("\n");
    }
    **/
    
    
    

    
    for (int i = 0 ; i < numCircles ; i++) {
        circleId = hitCirclesList[calcBlockId*numCircles + i];
        // if (indexX == 0 && indexY == 0) {
        //     //printf("circleId %d\t", circleId);          
        // }
        if (circleId < 0) {
            //printf("nCircles: %d\t\t", i);
            return;
        }
        int i3 = 3 * circleId;

        // Read position and radius
        float3 p = *(float3 *)(&cuConstRendererParams.position[i3]);

        shadePixel(pixelCenterNorm, p, imgPtr, circleId);

    }
    
    //printf("nCircles: N\t\t");

}

__global__ void kernelRenderPixelsLocal(int* hitCirclesList, int blockSize) {
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth; 
    int imageHeight = cuConstRendererParams.imageHeight; 

    if ((indexX >= imageWidth) || (indexY >= imageHeight))
        return;

    // Get pixelY, pixelX from index
    int pixelY = indexY;
    int pixelX = indexX;

    // For one pixel
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float4 *imagePtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float2 pixelCenter = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));

    //TODO: 

    // Get block number
    int gridXSize = (imageWidth + blockSize - 1) / blockSize;
    //blockId = rowNumber*width + columnNumber
    int blockId = ((pixelY/blockSize)*gridXSize) + (pixelX/blockSize);

    int calcBlockId = blockIdx.y*gridDim.x + blockIdx.x;
    calcBlockId = blockId;

    // Get list
    // Iterate over list of circles 
    int numCircles = cuConstRendererParams.numberOfCircles;
    int circleIndex; //

    float4 existingColor = *imagePtr;
    float4 newColor;
    for (int i = 0 ; i < numCircles ; i++) {
        circleIndex = hitCirclesList[calcBlockId*numCircles + i];
        // if (indexX == 0 && indexY == 0) {
        //     //printf("circleId %d\t", circleId);          
        // }
        if (circleIndex < 0) {
            //printf("nCircles: %d\t\t", i);
            break;
        }
        int i3 = 3 * circleIndex;

        // Read position and radius
        float3 p = *(float3 *)(&cuConstRendererParams.position[i3]);


        // shadePixel(pixelCenter, p, &localPtr, circleIndex);
        {
            float diffX = p.x - pixelCenter.x;
            float diffY = p.y - pixelCenter.y;
            float pixelDist = diffX * diffX + diffY * diffY;

            float rad = cuConstRendererParams.radius[circleIndex];
            float maxDist = rad * rad;

            // Circle does not contribute to the image
            if (pixelDist > maxDist)
                continue;

            float3 rgb;
            float alpha;

            // There is a non-zero contribution.  Now compute the shading value

            // Suggestion: This conditional is in the inner loop.  Although it
            // will evaluate the same for all threads, there is overhead in
            // setting up the lane masks, etc., to implement the conditional.  It
            // would be wise to perform this logic outside of the loops in
            // kernelRenderCircles.  (If feeling good about yourself, you
            // could use some specialized template magic).
            if (cuConstRendererParams.sceneName == SNOWFLAKES ||
                cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrt(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f - p.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

            } else {
                // Simple: each circle has an assigned color
                int index3 = 3 * circleIndex;
                rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
                alpha = .5f;
            }

            float oneMinusAlpha = 1.f - alpha;


            newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
            newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
            newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
            newColor.w = alpha + existingColor.w;

            // Local memory write
            existingColor = newColor;
        }
    }
    *imagePtr = newColor;
}


__global__ void kernelMarkBlocks(int blockSize, int* hitCircles, int numCircles) {
    // Parallellized over blocks and circles
    //TODO:
    float imageWidth = cuConstRendererParams.imageWidth;
    float imageHeight = cuConstRendererParams.imageHeight;

    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int circleIdx = threadIdx.x;

    int maxIterations = (numCircles + blockDim.x - 1) / blockDim.x;
    // Run mark on list by blockDim.x (1024) elements at a time.
    int j = 0;
    do {

        if (circleIdx >= numCircles)
            break;

        float3 p = *(float3 *)(&cuConstRendererParams.position[3 * (circleIdx)]);

        int gridXSize = (imageWidth + blockSize - 1) / blockSize;
        int blockX = blockIdx.x % gridXSize;
        int blockY = blockIdx.x / gridXSize;

        //CHANGE: Normalized and switched top and bottom?

        float boxL = (static_cast<float>(blockSize * blockX) + 0.5f)/imageWidth;
        float boxR = (static_cast<float>(blockSize * (blockX + 1)) + 0.5f)/imageWidth;
        float boxB = (static_cast<float>(blockSize * blockY) + 0.5f)/imageHeight;
        float boxT = (static_cast<float>(blockSize * (blockY + 1)) + 0.5f)/imageHeight;


        //CHANGE: Changes index to threadIdx.x

        // Mark blocks
        hitCircles[blockIdx.x*numCircles + circleIdx] = circleInBox(
            p.x, p.y,
            cuConstRendererParams.radius[circleIdx],
            boxL, boxR, boxT, boxB);

        circleIdx += blockDim.x;
        j++;
    } while (j < maxIterations);
}



__global__ void kernelScan2(int* hitCircles, int* prefixes, int numCircles) {
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;

    /**
    if (hitCircles[blockIdx.x*numCircles+2] == 1)
        printf("blockID: %d\t", blockIdx.x);
    **/
    
   
    /**
    //TODO: Hit circles don't seem to be marked correctly
    if (blockIdx.x == 2628 && threadIdx.x == 0) {
        printf("hitCircles: ");
        for (int j = 0 ; j < numCircles ; j++)
            printf("%d,", hitCircles[blockIdx.x*numCircles+j]);
        printf("\n");
    }
    **/
    
    

    //TODO loop over all scan blocks

    __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumScratch[2 * SCAN_BLOCK_DIM];


    // if (i == 0) {
    //     for (int j = 0; j < SCAN_BLOCK_DIM ; j++) {
    //         prefixSumInput[j] = 0;
    //         prefixSumOutput[j] = 0;
    //         prefixSumScratch[j] = 0;
    //         prefixSumScratch[SCAN_BLOCK_DIM+j] = 0;

    //     }
    // }

    // If number of circles / threads are less than block size, pad input array
    // Could spread if needed across threads instead of only last thread
    // Not sure if needed
    // if (i == 0 && numCircles < SCAN_BLOCK_DIM) {
    //     // printf("PAD");
    //     for (int j = numCircles ; j < SCAN_BLOCK_DIM ; j++) {
    //         prefixSumInput[j] = 0;
    //     }
    // }
    int maxIterations = (numCircles + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    //maxIterations = 1;
    // Run scan on list by SCAN_BLOCK_DIM elements at a time.
    int j = 0;
    int add;
    do {
        if (i < numCircles) {
            prefixSumInput[threadIdx.x] = hitCircles[blockIdx.x*numCircles + i];
        }
        __syncthreads();
        sharedMemExclusiveScan(threadIdx.x, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);
        __syncthreads();
        if (i < numCircles) {
            // prefix = currrent prefix output + final output of previous sum
            prefixes[blockIdx.x*numCircles + i] = prefixSumOutput[threadIdx.x];
            if (j >= 1) {
                add = prefixes[(blockIdx.x*numCircles) + ((j*SCAN_BLOCK_DIM)-1)];
                add = add >= 0 ? add : 0 ;
                if (hitCircles[(blockIdx.x*numCircles) + ((j*SCAN_BLOCK_DIM)-1)] == 1)
                    add += 1;
                prefixes[blockIdx.x*numCircles + i] += add;
            }
        }
        i += SCAN_BLOCK_DIM;
        j++;
    } while (j < maxIterations);
}


__global__ void kernelMakeLists(int* hitCircles, int* prefixes, int* hitCirclesList) {
    // Parallellized over blocks and list 
    //TODO:
    int numCircles = cuConstRendererParams.numberOfCircles;
    int circleIdx = threadIdx.x;
    /**
    // Prefix list seems correct
    if (blockIdx.x == 2646 && threadIdx.x == 0) {
        printf("prefixList: ");
        for (int j = 0 ; j < numCircles ; j++)
            printf("%d,", prefixes[blockIdx.x*length+j]);
        printf("\n");
    }
    **/
    // int blockIndex = blockIdx.x;
    int maxIterations = (numCircles + blockDim.x - 1) / blockDim.x;
    // Run makelist on list by blockDim.x (1024) elements at a time.
    // maxIterations = 1;
    int j = 0;
    int index;
    do {
        if (circleIdx >= numCircles)
            break;

        index = blockIdx.x * numCircles + circleIdx;
        int idx = prefixes[index];
        if (hitCircles[index] == 1) {
            hitCirclesList[blockIdx.x*numCircles + idx] = circleIdx;
        }

        
        // If last element and no of indices < no of circles
        if (circleIdx == numCircles - 1) {
            if (hitCircles[index] == 1 && idx < numCircles - 1)
                hitCirclesList[blockIdx.x*numCircles + idx+1] = -1; 
            else if (hitCircles[index] == 0)
                hitCirclesList[blockIdx.x*numCircles + idx] = -1;
            /**
            if (blockIdx.x == 2592) {
                printf("idx: %d, hc[idx] %d, hc[idx+1] %d\n", idx, hitCirclesList[blockIdx.x*blockDim.x + idx], hitCirclesList[blockIdx.x*blockDim.x + idx+1]);
            }
            **/
            
        }
        

        circleIdx += blockDim.x;
        j++;
    } while (j < maxIterations);

}

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

}

CudaRenderer::~CudaRenderer() {
    if (image) {
        delete image;
    }

    if (position) {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    if (cudaDevicePosition) {
        cudaCheckError( cudaFree(hitCircles));
        cudaCheckError( cudaFree(prefixes));
        cudaCheckError( cudaFree(hitCirclesList));
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image *CudaRenderer::getImage() {
    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data, cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void CudaRenderer::setup() {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0) {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU) {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    // TODO: Add data structures

    // Set numBlocks
    // TODO: CHANGED: 
    // numBlocks = (image->width*image->height)/(blockSize*blockSize); 
    int blocksXSize = (image->width + blockSize - 1) / blockSize; 
    int blocksYSize = (image->height + blockSize - 1) / blockSize; 

    numBlocks = blocksXSize * blocksYSize;    

    cudaCheckError( cudaMalloc(&hitCircles, numBlocks * sizeof(int) * numberOfCircles));
    cudaCheckError( cudaMalloc(&prefixes, numBlocks * sizeof(int) * numberOfCircles));
    cudaCheckError( cudaMalloc(&hitCirclesList, numBlocks * sizeof(int) * numberOfCircles));

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int *permX;
    int *permY;
    float *value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {.8f, .9f, 1.f}, {.8f, .9f, 1.f}, {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void CudaRenderer::allocOutputImage(int width, int height) {
    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x,
                 (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void CudaRenderer::advanceAnimation() {
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // Only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::render() {
    int numCircles = numberOfCircles;
   
    // CHANGE: Bad fix, set all to -1;
    // cudaCheckError(cudaMemset(hitCirclesList, -1, numBlocks*numCircles*sizeof(int)))
    // TODO:
    // Clear all data structures?
    // Think? Switch blocks and circles

    // Kernel to mark blocks as intersecting
    // Parallel over blocks and circles
    dim3 blockDim(1024, 1);
    dim3 gridDim(numBlocks);
    // printf("numCircles %d\n", numCircles);
    kernelMarkBlocks<<<gridDim, blockDim>>>(blockSize, hitCircles, numCircles);
    cudaCheckError( cudaDeviceSynchronize() );

    //Kernel to scan marked lists
    // Parallel over blocks
    // blockDim = dim3(1);
    // gridDim = dim3((numBlocks + blockDim.x - 1) / blockDim.x);
    // kernelScan<<<gridDim, blockDim>>>(hitCircles, prefixes);
    // thrust::device_ptr<int> hitC_ptr = thrust::device_pointer_cast(hitCircles);
    // thrust::device_ptr<int> prefixes_ptr = thrust::device_pointer_cast(prefixes);

    // for (int i = 0 ; i < numBlocks ; i++) {
    //     thrust::exclusive_scan(hitC_ptr + (i*numCircles), hitC_ptr + ((i+1)*numCircles), prefixes_ptr + (i*numCircles));

    //     // thrust::exclusive_scan(hitCircles + (i*numCircles), hitCircles + ((i+1)*numCircles), prefixes + (i*numCircles));
    // }

    //TODO: exclusiveScan Kernel
    blockDim = dim3(SCAN_BLOCK_DIM, 1);
    gridDim = dim3(numBlocks);
    kernelScan2<<<gridDim, blockDim>>>(hitCircles, prefixes, numCircles);
    cudaCheckError( cudaDeviceSynchronize() );


    //Kernel to make list of indices
    // Parallel over blocks and prefix list
    // TODO: Change numCircles to 1024
    // TODO: Change maxIterations to
    blockDim = dim3(1024, 1);
    gridDim = dim3(numBlocks);
    kernelMakeLists<<<gridDim, blockDim>>>(hitCircles, prefixes, hitCirclesList);
    cudaCheckError( cudaDeviceSynchronize() );

  
    // 256 threads per block is a healthy number
    int threadBlockSize = blockSize;
    threadBlockSize = 32;
    blockDim = dim3(threadBlockSize, threadBlockSize);
    //dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);
    gridDim = dim3((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);
    // 72x72
    //printf("dims: %d, %d\n", gridDim.x, gridDim.y);
    // Kernel to render pixels used data
    /**
    KernelRenderPixelsNew<<<>>>>   
    **/

    // kernelRenderPixels<<<gridDim, blockDim>>>();
    kernelRenderPixelsLocal<<<gridDim, blockDim>>>(hitCirclesList, blockSize);

    cudaCheckError( cudaDeviceSynchronize() );
    // cudaDeviceSynchronize();
}
