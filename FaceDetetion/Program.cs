using System;
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
                float harrisThreshold = 0.01f;
                float faceClusterThreshold = 0.1f;
                var corners = HarrisCorners(edgeImage, harrisThreshold);
                var faces = DetectFaces(corners, faceClusterThreshold * edgeImage.Width);
                Bitmap annotatedImage = AnnotateExpression(edgeImage, faces);
                annotatedImage.Save(outputFilePath);
                Console.WriteLine($"Edge and expression detection completed for {file}. Result saved to {outputFilePath}");
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
                int grayScale = (int)((originalColor.R * 0.3) + (originalColor.G * 0.59) + (originalColor.B * 0.11));
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

    private static List<Point> HarrisCorners(Bitmap image, float threshold)
    {
        int width = image.Width;
        int height = image.Height;
        List<Point> corners = new List<Point>();

        Func<int, int, int> getPixel = (x, y) =>
        {
            if (x < 0 || y < 0 || x >= width || y >= height) return 0;
            return image.GetPixel(x, y).R;
        };

        for (int y = 1; y < height - 2; y++)
        {
            for (int x = 1; x < width - 2; x++)
            {
                int ix = (getPixel(x + 1, y) - getPixel(x - 1, y)) / 2;
                int iy = (getPixel(x, y + 1) - getPixel(x, y - 1)) / 2;
                int ixx = ix * ix;
                int iyy = iy * iy;
                int ixy = ix * iy;
                int det = ixx * iyy - ixy * ixy;
                int trace = ixx + iyy;
                float r = det - 0.04f * (trace * trace);
                if (r > threshold)
                {
                    corners.Add(new Point(x, y));
                }
            }
        }
        return corners;
    }

    private static List<List<Point>> DetectFaces(List<Point> corners, float threshold)
    {
        List<List<Point>> clusters = new List<List<Point>>();

        Func<Point, Point, double> distance = (p1, p2) =>
        {
            return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
        };

        foreach (var corner in corners)
        {
            bool added = false;
            foreach (var cluster in clusters)
            {
                if (cluster.Exists(c => distance(c, corner) < threshold))
                {
                    cluster.Add(corner);
                    added = true;
                }
            }

            if (!added)
            {
                List<Point> newCluster = new List<Point> { corner };
                clusters.Add(newCluster);
            }
        }

        return clusters.FindAll(cluster => cluster.Count > 10);
    }

    private static string DetectExpression(Bitmap image, List<Point> face)
    {
        int minX = int.MaxValue;
        int minY = int.MaxValue;
        int maxX = int.MinValue;
        int maxY = int.MinValue;

        foreach (var point in face)
        {
            if (point.X < minX) minX = point.X;
            if (point.Y < minY) minY = point.Y;
            if (point.X > maxX) maxX = point.X;
            if (point.Y > maxY) maxY = point.Y;
        }

        Rectangle mouthRegion = new Rectangle(minX, minY + (maxY - minY) * 2 / 3, maxX - minX, (maxY - minY) / 3);
        Rectangle eyeRegion = new Rectangle(minX, minY, maxX - minX, (maxY - minY) / 3);

        Bitmap mouthImage = image.Clone(mouthRegion, image.PixelFormat);
        Bitmap eyeImage = image.Clone(eyeRegion, image.PixelFormat);

        Func<Bitmap, int> averageColor = (bmp) =>
        {
            int totalColor = 0;
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    Color color = bmp.GetPixel(x, y);
                    totalColor += (color.R + color.G + color.B) / 3;
                }
            }
            return totalColor / (bmp.Width * bmp.Height);
        };

        int mouthIntensity = averageColor(mouthImage);
        int eyeIntensity = averageColor(eyeImage);

        int smileThreshold = 100;
        int frownThreshold = 50;

        if (mouthIntensity > smileThreshold) return "smile";
        else if (mouthIntensity < frownThreshold) return "frown";
        else return "neutral";
    }

    private static Bitmap AnnotateExpression(Bitmap image, List<List<Point>> faces)
    {
        Bitmap annotatedImage = new Bitmap(image);
        using (Graphics graphics = Graphics.FromImage(annotatedImage))
        {
            foreach (var face in faces)
            {
                string expression = DetectExpression(image, face);
                int minX = int.MaxValue;
                int minY = int.MaxValue;

                foreach (var point in face)
                {
                    if (point.X < minX) minX = point.X;
                    if (point.Y < minY) minY = point.Y;
                }

                graphics.DrawString(expression, new Font("Arial", 16.0f), Brushes.Red, minX, minY);
            }
        }
        return annotatedImage;
    }
}
