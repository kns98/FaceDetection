using System;
using System.Collections.Generic;
using System.Drawing;

public class CornerFinder
{
    public static List<Point> FindCorners(Bitmap response, double threshold, int minDistance)
    {
        List<Point> corners = new List<Point>();
        FindCornersRecursive(response, threshold, minDistance, 0, 0, response.Width, response.Height, corners);
        return corners;
    }

    private static void FindCornersRecursive(Bitmap response, double threshold, int minDistance, int startX, int startY, int endX, int endY, List<Point> corners)
    {
        // If the area to process is small enough, process it directly
        if ((endX - startX) * (endY - startY) <= 100) // Adjust the threshold as needed
        {
            ProcessGrid(response, threshold, minDistance, startX, startY, endX, endY, corners);
            return;
        }

        // Otherwise, divide the area into four smaller areas
        int midX = (startX + endX) / 2;
        int midY = (startY + endY) / 2;

        FindCornersRecursive(response, threshold, minDistance, startX, startY, midX, midY, corners); // Top-left
        FindCornersRecursive(response, threshold, minDistance, midX, startY, endX, midY, corners); // Top-right
        FindCornersRecursive(response, threshold, minDistance, startX, midY, midX, endY, corners); // Bottom-left
        FindCornersRecursive(response, threshold, minDistance, midX, midY, endX, endY, corners); // Bottom-right
    }

    private static void ProcessGrid(Bitmap response, double threshold, int minDistance, int startX, int startY, int endX, int endY, List<Point> corners)
    {
        for (int y = startY; y < endY; y++)
        {
            for (int x = startX; x < endX; x++)
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
    }
}
