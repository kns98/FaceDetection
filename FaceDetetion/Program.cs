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

    private static void ProcessImageFile(object state)
    {
        string file = (string)state;
        string inputDirectoryPath = Path.GetDirectoryName(file);
        string outputFileName = "edge_" + Path.GetFileName(file);
        string outputFilePath = Path.Combine(inputDirectoryPath, outputFileName);

        try
        {
            using (Bitmap inputImage = new Bitmap(file))
            {
                Bitmap edgeImage = SobelEdgeDetection(inputImage);
                Bitmap harrisImage = DetectHarrisCorners(edgeImage);
                harrisImage.Save(outputFilePath);
                Console.WriteLine($"Edge and corner detection completed for {file}. Result saved to {outputFilePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred processing {file}: {ex.Message}");
        }
        finally
        {
            Interlocked.Decrement(ref processedFiles);
        }
    }

    private static Bitmap ConvertToGrayscale(Bitmap image)
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


    private static Bitmap DetectHarrisCorners(Bitmap image)
    {
        Bitmap grayscaleImage = ConvertToGrayscale(image);
        (Bitmap Ix, Bitmap Iy) = CalculateSpatialDerivatives(grayscaleImage);
        Bitmap responseImage = CalculateHarrisResponse(grayscaleImage, Ix, Iy);
        Bitmap corners = NonMaximumSuppression(responseImage);

        return corners;
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
        Bitmap Ix = new Bitmap(image.Width, image.Height);
        Bitmap Iy = new Bitmap(image.Width, image.Height);

        for (int y = 1; y < image.Height - 1; y++)
        {
            for (int x = 1; x < image.Width - 1; x++)
            {
                int dx = (image.GetPixel(x + 1, y).R - image.GetPixel(x - 1, y).R) / 2;
                int dy = (image.GetPixel(x, y + 1).R - image.GetPixel(x, y - 1).R) / 2;

                Ix.SetPixel(x, y, Color.FromArgb(dx, dx, dx));
                Iy.SetPixel(x, y, Color.FromArgb(dy, dy, dy));
            }
        }

        return (Ix, Iy);
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



    /*
     * Understanding Harris Corner Detection Algorithm in Image Processing
     *
     * Harris Corner Detection is a significant method in the field of computer vision and image processing,
     * primarily used to identify corners and interest points in an image. The code snippet provided implements
     * a function `CalculateHarrisResponse`, which calculates the Harris response for each pixel in a given image.
     * This response helps in detecting corners by analyzing the gradients in the image.
     *
     * Overview of Harris Corner Detection
     *
     * The Harris Corner Detection algorithm was developed by Chris Harris and Mike Stephens in 1988 and is
     * widely used for corner detection because of its accuracy and efficiency.
     *
     * The core idea of the Harris Corner Detection algorithm is based on the differential of the corner score
     * with respect to direction, which measures the changes in intensity. Corners are regions in the image
     * where the intensity changes significantly in all directions. The algorithm involves the following steps:
     *
     * 1. Compute the image gradients (Ix and Iy) using a derivative mask.
     * 2. Compute the products of derivatives at each pixel: Ix^2, Iy^2, and Ix*Iy.
     * 3. Apply a Gaussian window function to smooth the derivative products.
     * 4. Compute the corner response R for each pixel using the formula:
     *    R = det(M) - k * (trace(M))^2
     *    where M is the second moment matrix or the structure tensor at each pixel.
     *
     * Let's now break down the provided code to understand its implementation in C#.
     *
     * Code Breakdown
     *
     * Function Definition
     */
    private static Bitmap CalculateHarrisResponse(Bitmap image, Bitmap Ix, Bitmap Iy)
    {
        /*
         * The function `CalculateHarrisResponse` is defined as `private static`, indicating it is a static method
         * that cannot be accessed outside its class. It takes three parameters: the original image (`image`),
         * and the gradient images (`Ix` and `Iy`), which represent the gradients in the x and y directions, respectively.
         *
         * The method initializes a new bitmap `response` to store the Harris response values and a 2D array `responseValues`
         * to keep the calculated response values for each pixel. It also initializes `maxResponse` and `minResponse`
         * to track the maximum and minimum response values, which will be used for normalization later.
         */
        Bitmap response = new Bitmap(image.Width, image.Height);
        double[,] responseValues = new double[image.Width, image.Height];
        double maxResponse = double.MinValue;
        double minResponse = double.MaxValue;

        /*
         * Gradient Computation Loop
         *
         * This nested loop iterates through each pixel in the image, except for the border pixels. For each pixel,
         * it initializes `Ixx`, `Iyy`, and `Ixy` to zero. These variables will accumulate the products of gradients
         * over a 3x3 neighborhood around the pixel.
         *
         * Within the inner loops, the code fetches the gradient values `ix` and `iy` for each pixel in the 3x3 neighborhood
         * using the `GetPixel` method. It then computes the products of these gradients, which are summed into `Ixx`, `Iyy`,
         * and `Ixy`.
         */
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

                /*
                 * After exiting the inner loops, the code computes the determinant and trace of the second moment matrix M,
                 * and subsequently calculates the Harris response value for the pixel using the formula:
                 * R = det(M) - k * (trace(M))^2
                 * Here, k is an empirical constant, typically around 0.04. The response value is stored in `responseValues`,
                 * and `maxResponse` and `minResponse` are updated to keep track of the range of response values.
                 */
                double determinant = (Ixx * Iyy) - (Ixy * Ixy);
                double trace = Ixx + Iyy;
                double responseValue = determinant - 0.04 * trace * trace;

                responseValues[x, y] = responseValue;
                if (responseValue > maxResponse) maxResponse = responseValue;
                if (responseValue < minResponse) minResponse = responseValue;
            }
        }

        /*
         * Normalizing the Response Values
         *
         * After calculating the response values for all pixels, the code normalizes these values to the 0-255 range.
         * This normalization is essential for converting the response values into a grayscale image where the intensity
         * of each pixel represents the strength of the corner response.
         *
         * The normalization formula used here scales the response values linearly between the minimum and maximum response
         * values observed. The normalized value is then set as the intensity of the corresponding pixel in the `response` bitmap.
         */
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                int normalizedValue = (int)(255.0 * (responseValues[x, y] - minResponse) / (maxResponse - minResponse));
                response.SetPixel(x, y, Color.FromArgb(normalizedValue, normalizedValue, normalizedValue));
            }
        }

        /*
         * Returning the Result
         *
         * Finally, the function returns the `response` bitmap, which now contains the Harris corner response values as a grayscale image.
         */
        return response;
    }
}

/*
 * Practical Considerations
 *
 * When implementing the Harris Corner Detection algorithm, there are several practical considerations to keep in mind:
 *
 * 1. Gradient Computation: The gradients `Ix` and `Iy` can be computed using various methods, such as Sobel operators
 *    or simple finite differences. The choice of method affects the accuracy and efficiency of the algorithm.
 *
 * 2. Gaussian Smoothing: Applying a Gaussian filter to the gradient products `Ixx`, `Iyy`, and `Ixy` can improve the stability
 *    of the corner detection by reducing noise. This step is not included in the provided code but is a common enhancement.
 *
 * 3. Thresholding: After computing the Harris response, it is common to apply a threshold to identify significant corners.
 *    Pixels with response values above the threshold are considered corners.
 *
 * 4. Non-Maximum Suppression: To avoid detecting multiple corners in the vicinity of a single corner, non-maximum suppression
 *    can be applied. This step retains only the local maxima in the response image.
 *
 * Conclusion
 *
 * The provided code snippet effectively implements the core of the Harris Corner Detection algorithm by computing the Harris
 * response for each pixel in an image. It involves calculating the image gradients, summing the products of these gradients
 * over a neighborhood, and then using these sums to compute the Harris response. Finally, the response values are normalized
 * and stored in a bitmap.
 *
 * By understanding and extending this implementation, one can build robust corner detection functionality suitable for various
 * computer vision applications, such as image matching, object recognition, and motion tracking.
 */
