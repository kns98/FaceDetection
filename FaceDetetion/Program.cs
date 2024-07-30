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

    private static Bitmap SobelEdgeDetection(Bitmap image)
    {
        Bitmap grayScaleImage = ConvertToGrayscale(image);
        Bitmap edgeImage = new Bitmap(grayScaleImage.Width, grayScaleImage.Height);

        int[,] xKernel = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        int[,] yKernel = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

        for (int y = 1; y < grayScaleImage.Height - 1; y++)
        {
            for (int x = 1; x < grayScaleImage.Width - 1; x++)
            {
                float xGradient = 0;
                float yGradient = 0;

                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        Color pixel = grayScaleImage.GetPixel(x + kx, y + ky);
                        int pixelBrightness = pixel.R;
                        xGradient += pixelBrightness * xKernel[ky + 1, kx + 1];
                        yGradient += pixelBrightness * yKernel[ky + 1, kx + 1];
                    }
                }

                int gradientMagnitude = (int)Math.Min(Math.Sqrt(xGradient * xGradient + yGradient * yGradient), 255);
                edgeImage.SetPixel(x, y, Color.FromArgb(gradientMagnitude, gradientMagnitude, gradientMagnitude));
            }
        }

        return edgeImage;
    }

    private static Bitmap DetectHarrisCorners(Bitmap image)
    {
        Bitmap grayscaleImage = ConvertToGrayscale(image);
        (Bitmap Ix, Bitmap Iy) = CalculateSpatialDerivatives(grayscaleImage);
        Bitmap responseImage = CalculateHarrisResponse(grayscaleImage, Ix, Iy);
        Bitmap corners = NonMaximumSuppression(responseImage);

        return corners;
    }

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

    private static Bitmap CalculateHarrisResponse(Bitmap image, Bitmap Ix, Bitmap Iy)
    {
        Bitmap response = new Bitmap(image.Width, image.Height);

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

                int r = Math.Max(0, Math.Min(255, (int)responseValue));
                response.SetPixel(x, y, Color.FromArgb(r, r, r));
            }
        }

        return response;
    }

    private static Bitmap NonMaximumSuppression(Bitmap response)
    {
        Bitmap corners = new Bitmap(response.Width, response.Height);

        for (int y = 1; y < response.Height - 1; y++)
        {
            for (int x = 1; x < response.Width - 1; x++)
            {
                Color pixelColor = response.GetPixel(x, y);
                bool isCorner = true;

                for (int v = -1; v <= 1; v++)
                {
                    for (int u = -1; u <= 1; u++)
                    {
                        if (u == 0 && v == 0) continue;
                        if (response.GetPixel(x + u, y + v).R > pixelColor.R)
                        {
                            isCorner = false;
                            break;
                        }
                    }
                    if (!isCorner) break;
                }

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

        return corners;
    }
}
