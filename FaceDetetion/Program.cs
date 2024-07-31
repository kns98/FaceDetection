/**
 * Overview of Edge Detection and Harris Corner Detection in Face Detection
 *
 * Introduction:
 * Edge detection and corner detection are fundamental techniques in image processing and computer vision,
 * used to extract meaningful features from images. In the context of face detection, Sobel edge detection
 * and Harris corner detection are employed to identify and localize faces within images. This section
 * details how these algorithms are applied specifically to face detection, exploring their significance,
 * methodologies, and the impact they have on enhancing the accuracy and reliability of face detection systems.
 */

/**
 * Importance of Edge Detection:
 *
 * 1. Structural Information Extraction:
 *    Edges represent significant local changes in image intensity, corresponding to object boundaries.
 *    Detecting these edges extracts critical structural information essential for tasks like object
 *    recognition, image segmentation, and face detection. The accurate identification of edges helps in
 *    delineating the boundaries of facial features, providing a clear structure that is vital for further
 *    processing in face detection algorithms.
 *
 * 2. Simplicity and Efficiency:
 *    The Sobel edge detection algorithm is simple and computationally efficient, making it suitable for
 *    real-time applications where processing speed is crucial. It uses convolution with Sobel kernels to
 *    compute the image intensity gradient at each pixel. The straightforward nature of the Sobel operator
 *    allows for quick implementation and execution, which is especially important in systems requiring
 *    fast processing times, such as live video analysis or real-time facial recognition systems.
 *
 * 3. Gradient Information:
 *    Sobel edge detection provides gradient information in the x and y directions, valuable for
 *    understanding the orientation and magnitude of edges. This information is further utilized in more
 *    complex image analysis tasks, such as contour detection and feature extraction. By analyzing the
 *    gradient magnitude and direction, it is possible to obtain a detailed representation of the image
 *    structure, which is crucial for accurate face detection.
 */

/**
 * Advantages of Harris Corner Detection:
 *
 * 1. Corner Detection Robustness:
 *    Harris corner detection is robust in identifying corners, which are points where intensity changes
 *    significantly in multiple directions. Corners are vital for image matching, tracking, and 3D
 *    reconstruction as they are more stable under various transformations. In the context of face detection,
 *    corners correspond to key facial landmarks, enhancing the precision of face localization. The stability
 *    of corners under different imaging conditions makes them reliable features for consistent face detection.
 *
 * 2. Mathematical Foundation:
 *    Harris corner detection is based on second-order derivatives of the image, using the structure tensor
 *    or second moment matrix. This allows accurate corner detection even in the presence of noise and minor
 *    image distortions. The mathematical rigor behind the Harris corner detector ensures that it can
 *    effectively identify corners in varying image conditions, contributing to the robustness and accuracy
 *    of face detection algorithms.
 *
 * 3. Non-Maximum Suppression:
 *    The method includes a non-maximum suppression step, refining detected corners by eliminating weaker
 *    responses and keeping only the strongest points. This results in a cleaner and more precise set of
 *    feature points. The non-maximum suppression step is critical for reducing false positives and ensuring
 *    that the detected corners correspond to actual significant features in the image, thereby enhancing the
 *    reliability of face detection systems.
 */

/**
 * Application in Face Detection:
 *
 * 1. Preprocessing:
 *    - Convert the input image to grayscale to simplify processing and reduce the image to a single channel
 *      containing sufficient information for detecting facial features. Grayscale conversion reduces the
 *      complexity of the image data, making it easier to analyze and process while retaining the essential
 *      details needed for effective face detection.
 *
 * 2. Edge Detection:
 *    - Apply the Sobel operator to the grayscale image to calculate the gradient magnitude at each pixel,
 *      highlighting regions with significant intensity changes, typically corresponding to edges in the
 *      image. Sobel edge detection uses two 3x3 convolution kernels to detect edges in the horizontal (x) and
 *      vertical (y) directions. The gradient magnitudes are calculated as follows:
 *      - G_x = Sobel Kernel_x * Image
 *      - G_y = Sobel Kernel_y * Image
 *      - G = sqrt(G_x^2 + G_y^2)
 *    - This results in an edge map highlighting boundaries of facial features like eyes, nose, and mouth.
 *      The edge map provides a preliminary structure that guides further analysis and feature extraction
 *      processes.
 *
 * 3. Edge Map Analysis:
 *    - Analyze the edge map to identify contours of potential face regions. By focusing on edge density and
 *      arrangement, the algorithm localizes regions resembling a human face structure. Facial features
 *      typically have distinct edge patterns: eyes form dark regions with strong edges around them, and the
 *      mouth forms a horizontal edge with a significant gradient. This analysis helps in narrowing down
 *      the regions of interest in the image, making the subsequent feature detection steps more efficient.
 *
 * 4. Feature Point Detection:
 *    - After edge detection, apply Harris corner detection to identify key feature points within potential
 *      face regions. These points correspond to corners and intersections characteristic of facial landmarks.
 *      Compute the gradient products and use the structure tensor to determine the corner response for each
 *      pixel:
 *      - M = sum_w [ I_x^2, I_x I_y ]
 *               [ I_x I_y, I_y^2 ]
 *      - R = det(M) - k * (trace(M))^2
 *    - Corners are identified where the response R is above a certain threshold, indicating significant
 *      changes in intensity in multiple directions. The identified corners provide a set of candidate points
 *      for facial landmarks, which are essential for accurate face detection.
 *
 * 5. Identifying Facial Landmarks:
 *    - Analyze detected corners to identify facial landmarks like the corners of the eyes, mouth, and tip of
 *      the nose. By comparing the spatial arrangement and relative positions of these corners, the algorithm
 *      validates the presence of a face and pinpoints its exact location. The spatial analysis ensures that
 *      the detected features align with the expected configuration of facial features, enhancing the accuracy
 *      of face detection.
 *
 * 6. Clustering and Region Proposal:
 *    - Cluster detected corners to group them into regions likely corresponding to faces. Use clustering
 *      algorithms like DBSCAN or k-means based on proximity. Clustering helps in organizing the detected
 *      corners into coherent regions, each representing a potential face. Evaluate each cluster to confirm
 *      it matches the expected configuration of facial features, eliminating false positives and ensuring
 *      accurate representation of faces. This step further refines the face detection process by focusing on
 *      regions with a high probability of containing a face.
 */

/**
 * Combined Application in Face Detection Pipeline:
 *
 * 1. Image Preprocessing:
 *    - Resize and normalize the input image to ensure consistent processing. Convert to grayscale and apply
 *      histogram equalization to enhance contrast. These preprocessing steps improve the quality of the
 *      input image, making it more suitable for subsequent edge and corner detection algorithms.
 *
 * 2. Edge and Corner Detection:
 *    - Apply Sobel edge detection to generate an edge map highlighting potential facial boundaries. The edge
 *      map provides a preliminary structure that guides further analysis. Apply Harris corner detection to
 *      the edge map to identify key feature points within potential face regions. The combination of edge
 *      and corner detection ensures that both the boundaries and key features of the face are accurately
 *      identified.
 *
 * 3. Feature Extraction:
 *    - Extract features like edge density, corner clusters, and geometric arrangements from the edge map and
 *      corner points. These features form a feature vector representing each potential face region. The
 *      feature vector encapsulates the essential information needed for accurate face detection and
 *      classification.
 *
 * 4. Face Region Proposal:
 *    - Use extracted features to propose candidate face regions. Evaluate each region based on the spatial
 *      arrangement of edges and corners to ensure it matches the typical structure of a face. The evaluation
 *      process helps in filtering out non-face regions, improving the precision of face detection.
 *
 * 5. Validation and Refinement:
 *    - Validate proposed face regions using additional criteria like aspect ratio, symmetry, and relative
 *      distances between key feature points. Refine detected regions by applying non-maximum suppression to
 *      eliminate overlapping detections and improve accuracy. The validation and refinement steps ensure that
 *      the detected faces meet the expected characteristics of a human face, thereby enhancing the overall
 *      reliability of the face detection system.
 */

/**
 * Conclusion:
 * The combination of Sobel edge detection and Harris corner detection provides a robust approach to face
 * detection. Sobel edge detection highlights facial feature boundaries, while Harris corner detection
 * identifies key landmarks. Together, these methods enable accurate localization and identification of faces
 * in images, forming a critical component of modern face detection systems. By leveraging the strengths of
 * both algorithms, face detection systems can achieve high levels of accuracy and reliability, making them
 * suitable for a wide range of applications, from security systems to user authentication in mobile devices.
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
        string outputFilePath2 = Path.Combine(inputDirectoryPath, outputFileName2);
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

    private static Bitmap DetectHarrisCorners(Bitmap image)
    {
        try
        {
            Bitmap grayscaleImage = ConvertToGrayscale(image);
            (Bitmap Ix, Bitmap Iy) = CalculateSpatialDerivatives(grayscaleImage);
            Bitmap responseImage = CalculateHarrisResponse(grayscaleImage, Ix, Iy);
            //Bitmap corners = NonMaximumSuppression(responseImage);
            List<Point> corners_ = FindCorners(responseImage, 10, 10);

            // Draw circles around detected corners
            foreach (var corner in corners_)
            {
                DrawCircle(grayscaleImage, corner.X, corner.Y, 5, Color.Red);
            }

            return grayscaleImage;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred in DetectHarrisCorners: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }
    private static List<Point> FindCorners(Bitmap response, double threshold, int minDistance)
    {
        List<Point> corners = new List<Point>();
        for (int y = 1; y < response.Height - 1; y++)
        {
            for (int x = 1; x < response.Width - 1; x++)
            {
                if (response.GetPixel(x, y).R > threshold)
                {
                    bool tooClose = false;
                    foreach (var corner in corners)
                    {
                        int dx = corner.X - x;
                        int dy = corner.Y - y;
                        if (dx * dx + dy * dy < minDistance * minDistance)
                        {
                            tooClose = true;
                            break;
                        }
                    }
                    if (!tooClose)
                    {
                        corners.Add(new Point(x, y));
                    }
                }
            }
        }
        return corners;
    }

    private static void DrawCircle(Bitmap image, int centerX, int centerY, int radius, Color color)
    {
        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                if (x * x + y * y <= radius * radius)
                {
                    int drawX = centerX + x;
                    int drawY = centerY + y;
                    if (drawX >= 0 && drawX < image.Width && drawY >= 0 && drawY < image.Height)
                    {
                        image.SetPixel(drawX, drawY, color);
                    }
                }
            }
        }
    }

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
