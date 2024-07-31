﻿/**
 * Overview of Edge Detection and Harris Corner Detection in Face Detection
 *
 * Introduction:
 * Edge detection and corner detection are fundamental techniques in image processing and computer vision, 
 * used to extract meaningful features from images. In the context of face detection, Sobel edge detection 
 * and Harris corner detection are employed to identify and localize faces within images. This section 
 * details how these algorithms are applied specifically to face detection.
 *
 * Importance of Edge Detection:
 * 
 * 1. Structural Information Extraction:
 *    Edges represent significant local changes in image intensity, corresponding to object boundaries. 
 *    Detecting these edges extracts critical structural information essential for tasks like object 
 *    recognition, image segmentation, and face detection.
 *
 * 2. Simplicity and Efficiency:
 *    The Sobel edge detection algorithm is simple and computationally efficient, making it suitable for 
 *    real-time applications where processing speed is crucial. It uses convolution with Sobel kernels to 
 *    compute the image intensity gradient at each pixel.
 *
 * 3. Gradient Information:
 *    Sobel edge detection provides gradient information in the x and y directions, valuable for 
 *    understanding the orientation and magnitude of edges, further used in more complex image analysis 
 *    tasks.
 *
 * Advantages of Harris Corner Detection:
 * 
 * 1. Corner Detection Robustness:
 *    Harris corner detection is robust in identifying corners, which are points where intensity changes 
 *    significantly in multiple directions. Corners are vital for image matching, tracking, and 3D 
 *    reconstruction as they are more stable under various transformations.
 *
 * 2. Mathematical Foundation:
 *    Harris corner detection is based on second-order derivatives of the image, using the structure tensor 
 *    or second moment matrix. This allows accurate corner detection even in the presence of noise and minor 
 *    image distortions.
 *
 * 3. Non-Maximum Suppression:
 *    The method includes a non-maximum suppression step, refining detected corners by eliminating weaker 
 *    responses and keeping only the strongest points, resulting in a cleaner and more precise set of 
 *    feature points.
 *
 * Application in Face Detection:
 *
 * 1. Preprocessing:
 *    - Convert the input image to grayscale to simplify processing and reduce the image to a single channel 
 *      containing sufficient information for detecting facial features.
 *
 * 2. Edge Detection:
 *    - Apply the Sobel operator to the grayscale image to calculate the gradient magnitude at each pixel, 
 *      highlighting regions with significant intensity changes, typically corresponding to edges in the 
 *      image.
 *    - Sobel edge detection uses two 3x3 convolution kernels to detect edges in the horizontal (x) and 
 *      vertical (y) directions. The gradient magnitudes are calculated as follows:
 *      - G_x = Sobel Kernel_x * Image
 *      - G_y = Sobel Kernel_y * Image
 *      - G = sqrt(G_x^2 + G_y^2)
 *    - This results in an edge map highlighting boundaries of facial features like eyes, nose, and mouth.
 *
 * 3. Edge Map Analysis:
 *    - Analyze the edge map to identify contours of potential face regions. By focusing on edge density and 
 *      arrangement, the algorithm localizes regions resembling a human face structure.
 *    - Facial features typically have distinct edge patterns: eyes form dark regions with strong edges 
 *      around them, and the mouth forms a horizontal edge with a significant gradient.
 *
 * 4. Feature Point Detection:
 *    - After edge detection, apply Harris corner detection to identify key feature points within potential 
 *      face regions. These points correspond to corners and intersections characteristic of facial landmarks.
 *    - Compute the gradient products and use the structure tensor to determine the corner response for each 
 *      pixel:
 *      - M = sum_w [ I_x^2, I_x I_y ]
 *               [ I_x I_y, I_y^2 ]
 *      - R = det(M) - k * (trace(M))^2
 *    - Corners are identified where the response R is above a certain threshold, indicating significant 
 *      changes in intensity in multiple directions.
 *
 * 5. Identifying Facial Landmarks:
 *    - Analyze detected corners to identify facial landmarks like the corners of the eyes, mouth, and tip of 
 *      the nose. By comparing the spatial arrangement and relative positions of these corners, the algorithm 
 *      validates the presence of a face and pinpoints its exact location.
 *
 * 6. Clustering and Region Proposal:
 *    - Cluster detected corners to group them into regions likely corresponding to faces. Use clustering 
 *      algorithms like DBSCAN or k-means based on proximity.
 *    - Evaluate each cluster to confirm it matches the expected configuration of facial features, 
 *      eliminating false positives and ensuring accurate representation of faces.
 *
 * Combined Application in Face Detection Pipeline:
 * 
 * 1. Image Preprocessing:
 *    - Resize and normalize the input image to ensure consistent processing. Convert to grayscale and apply 
 *      histogram equalization to enhance contrast.
 *
 * 2. Edge and Corner Detection:
 *    - Apply Sobel edge detection to generate an edge map highlighting potential facial boundaries.
 *    - Apply Harris corner detection to the edge map to identify key feature points within potential face 
 *      regions.
 *
 * 3. Feature Extraction:
 *    - Extract features like edge density, corner clusters, and geometric arrangements from the edge map and 
 *      corner points. These form a feature vector representing each potential face region.
 *
 * 4. Face Region Proposal:
 *    - Use extracted features to propose candidate face regions. Evaluate each region based on the spatial 
 *      arrangement of edges and corners to ensure it matches the typical structure of a face.
 *
 * 5. Validation and Refinement:
 *    - Validate proposed face regions using additional criteria like aspect ratio, symmetry, and relative 
 *      distances between key feature points. Refine detected regions by applying non-maximum suppression to 
 *      eliminate overlapping detections and improve accuracy.
 *
 * Conclusion:
 * The combination of Sobel edge detection and Harris corner detection provides a robust approach to face 
 * detection. Sobel edge detection highlights facial feature boundaries, while Harris corner detection 
 * identifies key landmarks. Together, these methods enable accurate localization and identification of faces 
 * in images, forming a critical component of modern face detection systems.
 */
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading;

class EdgeDetectionApp
{
    static int processedFiles = 0;

    static void Main(string[] args)
    {
        try
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: EdgeDetectionApp <input directory path>");
                return;
            }

            string inputDirectoryPath = args[0];
            string[] extensions = { ".jpg", ".jpeg", ".png" };
            string[] files = Directory.GetFiles(inputDirectoryPath);

            foreach (string file in files)
            {
                if (Array.Exists(extensions, ext => file.EndsWith(ext, StringComparison.OrdinalIgnoreCase)))
                {
                    Interlocked.Increment(ref processedFiles);
                    ThreadPool.QueueUserWorkItem(ProcessImageFile, file);
                }
            }

            while (processedFiles > 0)
            {
                Thread.Sleep(100);
            }

            Console.WriteLine("All files have been processed.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in Main: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    private static void ProcessImageFile(object state)
    {
        string file = (string)state;
        string inputDirectoryPath = Path.GetDirectoryName(file);
        string outputFileName = "sobel_" + Path.GetFileName(file);
        string outputFileName2 = "harris_" + Path.GetFileName(file);
        string outputFilePath = Path.Combine(inputDirectoryPath, outputFileName);

        try
        {
            using (Bitmap inputImage = new Bitmap(file))
            {
                Bitmap edgeImage = SobelEdgeDetection(inputImage); 
                edgeImage.Save(outputFilePath);
                Bitmap harrisImage = DetectHarrisCorners(edgeImage);
                harrisImage.Save(outputFilePath2);
                Console.WriteLine($"Edge and corner detection completed for {file}. Result saved to {outputFilePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred processing {file}: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
        finally
        {
            Interlocked.Decrement(ref processedFiles);
        }
    }

    /**
     * This method converts a given Bitmap image to grayscale.
     * It iterates through each pixel, calculates the grayscale value using the luminance formula, 
     * and sets the pixel in the new grayscale image.
     *
     * @param image The input Bitmap image to be converted.
     * @return A new Bitmap image in grayscale.
     */
    private static Bitmap ConvertToGrayscale(Bitmap image)
    {
        try
        {
            Bitmap grayscale = new Bitmap(image.Width, image.Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color originalColor = image.GetPixel(x, y);
                    int grayScale = (int)(originalColor.R * 0.299 + originalColor.G * 0.587 + originalColor.B * 0.114);
                    Color grayColor = Color.FromArgb(grayScale, grayScale, grayScale);
                    grayscale.SetPixel(x, y, grayColor);
                }
            }
            return grayscale;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in ConvertToGrayscale: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw; // Re-throw the exception to allow higher-level handling if necessary
        }
    }

    /**
     * This method applies the Sobel edge detection algorithm to a given Bitmap image.
     * It first converts the image to grayscale, then calculates the gradient magnitude
     * at each pixel using Sobel kernels for both the x and y directions. The result is 
     * an edge-detected image.
     *
     * @param image The input Bitmap image to be processed.
     * @return A new Bitmap image containing the edges detected in the input image.
     */
    private static Bitmap SobelEdgeDetection(Bitmap image)
    {
        try
        {
            // Convert the image to grayscale for simplicity in processing
            Bitmap grayScaleImage = ConvertToGrayscale(image);
            Bitmap edgeImage = new Bitmap(grayScaleImage.Width, grayScaleImage.Height);

            // Define Sobel kernels for edge detection in the x and y directions
            int[,] xKernel = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            int[,] yKernel = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            // Loop through each pixel in the grayscale image, excluding the border pixels
            for (int y = 1; y < grayScaleImage.Height - 1; y++)
            {
                for (int x = 1; x < grayScaleImage.Width - 1; x++)
                {
                    float xGradient = 0;
                    float yGradient = 0;

                    // Apply the Sobel kernels to the surrounding pixels
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            Color pixel = grayScaleImage.GetPixel(x + kx, y + ky);
                            int pixelBrightness = pixel.R; // Use the red channel value as brightness
                            xGradient += pixelBrightness * xKernel[ky + 1, kx + 1];
                            yGradient += pixelBrightness * yKernel[ky + 1, kx + 1];
                        }
                    }

                    // Calculate the gradient magnitude and clamp it to the range [0, 255]
                    int gradientMagnitude = (int)Math.Min(Math.Sqrt(xGradient * xGradient + yGradient * yGradient), 255);

                    // Set the pixel in the edge image to the gradient magnitude
                    edgeImage.SetPixel(x, y, Color.FromArgb(gradientMagnitude, gradientMagnitude, gradientMagnitude));
                }
            }

            return edgeImage; // Return the processed edge-detected image
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in SobelEdgeDetection: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }

    /**
     * This method detects Harris corners in a given Bitmap image.
     * It converts the image to grayscale, calculates the spatial derivatives, 
     * computes the Harris response, and applies non-maximum suppression to 
     * identify corners.
     *
     * @param image The input Bitmap image in which to detect corners.
     * @return A new Bitmap image with detected corners highlighted in red.
     */
    private static Bitmap DetectHarrisCorners(Bitmap image)
    {
        try
        {
            Bitmap grayscaleImage = ConvertToGrayscale(image);
            (Bitmap Ix, Bitmap Iy) = CalculateSpatialDerivatives(grayscaleImage);
            Bitmap responseImage = CalculateHarrisResponse(grayscaleImage, Ix, Iy);
            Bitmap corners = NonMaximumSuppression(responseImage);

            return corners;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in DetectHarrisCorners: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }

    /**
     * This method calculates the spatial derivatives (Ix, Iy) of a given image using a simple central difference method.
     * Spatial derivatives are used to measure changes in intensity values along the x and y directions in the image.
     * 
     * The method returns a tuple containing two Bitmap objects: 
     * 1. Ix - The derivative of the image along the x-axis.
     * 2. Iy - The derivative of the image along the y-axis.
     * 
     * The central difference method is applied to each pixel (excluding the border pixels) to calculate the derivatives.
     * For each pixel (x, y) in the image, the derivative in the x direction (Ix) is computed as the difference between the 
     * right and left neighboring pixel values, divided by 2. Similarly, the derivative in the y direction (Iy) is computed 
     * as the difference between the top and bottom neighboring pixel values, divided by 2.
     * 
     * Parameters:
     *   Bitmap image - The input image for which the spatial derivatives are to be calculated.
     * 
     * Returns:
     *   (Bitmap, Bitmap) - A tuple containing the Ix and Iy derivative images.
     * 
     * Example usage:
     *   Bitmap inputImage = new Bitmap("path_to_image");
     *   (Bitmap Ix, Bitmap Iy) = CalculateSpatialDerivatives(inputImage);
     * 
     * Note:
     *   - The method assumes that the input image is a grayscale image. If the input image is in color, only the red channel
     *     is considered for derivative calculation.
     *   - The border pixels (the first and last rows and columns) are not processed to avoid boundary issues.
     */
    private static (Bitmap, Bitmap) CalculateSpatialDerivatives(Bitmap image)
    {
        try
        {
            Bitmap Ix = new Bitmap(image.Width, image.Height);
            Bitmap Iy = new Bitmap(image.Width, image.Height);
            double minIx = double.MaxValue, maxIx = double.MinValue;
            double minIy = double.MaxValue, maxIy = double.MinValue;

            // First pass: calculate gradients and find min and max values
            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    int dx = (image.GetPixel(x + 1, y).R - image.GetPixel(x - 1, y).R) / 2;
                    int dy = (image.GetPixel(x, y + 1).R - image.GetPixel(x, y - 1).R) / 2;

                    if (dx < minIx) minIx = dx;
                    if (dx > maxIx) maxIx = dx;
                    if (dy < minIy) minIy = dy;
                    if (dy > maxIy) maxIy = dy;
                }
            }

            // Second pass: normalize gradients and set pixels
            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    int dx = (image.GetPixel(x + 1, y).R - image.GetPixel(x - 1, y).R) / 2;
                    int dy = (image.GetPixel(x, y + 1).R - image.GetPixel(x, y - 1).R) / 2;

                    int normalizedDx = (int)(255.0 * (dx - minIx) / (maxIx - minIx));
                    int normalizedDy = (int)(255.0 * (dy - minIy) / (maxIy - minIy));

                    Ix.SetPixel(x, y, Color.FromArgb(normalizedDx, normalizedDx, normalizedDx));
                    Iy.SetPixel(x, y, Color.FromArgb(normalizedDy, normalizedDy, normalizedDy));
                }
            }

            return (Ix, Iy);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in CalculateSpatialDerivatives: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }


    /**
     * This method calculates the Harris response for each pixel in a given image using its spatial derivatives.
     * The Harris response is a measure of corner strength, computed using the determinant and trace of the second moment matrix.
     * 
     * The method returns a new Bitmap image where each pixel intensity represents the normalized Harris response value.
     * 
     * Parameters:
     *   Bitmap image - The input grayscale image.
     *   Bitmap Ix - The derivative of the image along the x-axis.
     *   Bitmap Iy - The derivative of the image along the y-axis.
     * 
     * Returns:
     *   Bitmap - A new Bitmap image with Harris response values normalized to the range [0, 255].
     * 
     * Note:
     *   - The method assumes that the input image and its derivatives are all grayscale images.
     *   - The border pixels (the first and last rows and columns) are not processed to avoid boundary issues.
     */
    private static Bitmap CalculateHarrisResponse(Bitmap image, Bitmap Ix, Bitmap Iy)
    {
        try
        {
            Bitmap response = new Bitmap(image.Width, image.Height);
            double[,] responseValues = new double[image.Width, image.Height];
            double maxResponse = double.MinValue;
            double minResponse = double.MaxValue;

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    double Ixx = 0, Iyy = 0, Ixy = 0;

                    for (int v = -1; v <= 1; v++)
                    {
                        for (int u = -1; u <= 1; u++)
                        {
                            int ix = Ix.GetPixel(x + u, y + v).R;
                            int iy = Iy.GetPixel(x + u, y + v).R;

                            Ixx += ix * ix;
                            Iyy += iy * iy;
                            Ixy += ix * iy;
                        }
                    }

                    double determinant = (Ixx * Iyy) - (Ixy * Ixy);
                    double trace = Ixx + Iyy;
                    double responseValue = determinant - 0.04 * trace * trace;

                    responseValues[x, y] = responseValue;
                    if (responseValue > maxResponse) maxResponse = responseValue;
                    if (responseValue < minResponse) minResponse = responseValue;
                }
            }

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    double normalizedResponse = 255.0 * (responseValues[x, y] - minResponse) / (maxResponse - minResponse);
                    int intensity = (int)normalizedResponse;
                    intensity = Math.Max(0, Math.Min(255, intensity));
                    response.SetPixel(x, y, Color.FromArgb(intensity, intensity, intensity));
                }
            }

            return response;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in CalculateHarrisResponse: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }

    /**
     * This method performs non-maximum suppression on a given Bitmap image.
     * Non-maximum suppression is a technique used to identify and keep the maximum response in a local neighborhood
     * while suppressing (setting to zero) all other responses. This is typically used in edge detection algorithms 
     * or corner detection algorithms to thin out the detected edges or corners.
     *
     * @param response The input Bitmap image on which non-maximum suppression is to be performed.
     * @return A Bitmap image where detected corners are highlighted in red, and non-corner pixels are set to black.
     */
    private static Bitmap NonMaximumSuppression(Bitmap response)
    {
        try
        {
            // Create a new Bitmap to store the result of the non-maximum suppression
            Bitmap corners = new Bitmap(response.Width, response.Height);

            // Loop through each pixel in the image, excluding the border pixels
            for (int y = 1; y < response.Height - 1; y++)
            {
                for (int x = 1; x < response.Width - 1; x++)
                {
                    // Get the color of the current pixel
                    Color pixelColor = response.GetPixel(x, y);
                    bool isCorner = true;

                    // Check the 3x3 neighborhood of the current pixel
                    for (int v = -1; v <= 1; v++)
                    {
                        for (int u = -1; u <= 1; u++)
                        {
                            // Skip the current pixel itself
                            if (u == 0 && v == 0) continue;
                            // If any neighboring pixel has a higher intensity, mark the current pixel as not a corner
                            if (response.GetPixel(x + u, y + v).R > pixelColor.R)
                            {
                                isCorner = false;
                                break;
                            }
                        }
                        // If the current pixel is not a corner, break out of the outer loop as well
                        if (!isCorner) break;
                    }

                    // If the current pixel is a corner, set it to red in the corners Bitmap, otherwise set it to black
                    if (isCorner)
                    {
                        corners.SetPixel(x, y, Color.Red);
                    }
                    else
                    {
                        corners.SetPixel(x, y, Color.Black);
                    }
                }
            }

            // Return the resulting Bitmap with the detected corners
            return corners;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in NonMaximumSuppression: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }
}
