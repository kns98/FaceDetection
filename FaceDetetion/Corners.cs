using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.IO;

public class CornerFinder
{
    private static int _imageCount = 0;

    public static List<Point> FindCorners(Bitmap response, double threshold, int minDistance)
    {
        var corners = new List<Point>();
        var geolocateDict = new GeolocateDict(minDistance);
        ProcessGrid(response, threshold, minDistance, 0, 0, response.Width, response.Height, corners, geolocateDict);
        return corners;
    }

    private static void ProcessGrid(Bitmap response, double threshold, int minDistance, int startX, int startY, int endX, int endY, List<Point> corners, GeolocateDict geolocateDict)
    {
        for (int y = startY; y < endY; y++)
        {
            for (int x = startX; x < endX; x++)
            {
                if (response.GetPixel(x, y).R > threshold)
                {
                    if (!geolocateDict.IsTooClose(x, y))
                    {
                        corners.Add(new Point(x, y));
                        geolocateDict.AddPoint(x, y);
                    }
                }
            }

            if ((y - startY) % 1000 == 0) // Save partial image every 1000 rows
            {
                SavePartialImage(response, startX, startY, endX, y);
            }
        }
    }

    private static void SavePartialImage(Bitmap image, int startX, int startY, int endX, int endY)
    {
        string fileName = $"partial_image_{_imageCount}.png";
        using (Bitmap partialImage = new Bitmap(endX - startX, endY - startY))
        {
            using (Graphics g = Graphics.FromImage(partialImage))
            {
                g.DrawImage(image, 0, 0, new Rectangle(startX, startY, endX - startX, endY - startY), GraphicsUnit.Pixel);
            }
            partialImage.Save(fileName, ImageFormat.Png);
        }
        _imageCount++;
    }
}

public class GeolocateDict
{
    private readonly int _minDistance;
    private readonly Dictionary<int, List<Point>> _dict;
    private int _lookupCount;
    private readonly Stopwatch _stopwatch;

    public GeolocateDict(int minDistance)
    {
        _minDistance = minDistance;
        _dict = new Dictionary<int, List<Point>>();
        _lookupCount = 0;
        _stopwatch = new Stopwatch();
        _stopwatch.Start();
    }

    public void AddPoint(int x, int y)
    {
        int hash = GetHash(x, y);
        if (!_dict.ContainsKey(hash))
        {
            _dict[hash] = new List<Point>();
        }
        _dict[hash].Add(new Point(x, y));
    }

    public bool IsTooClose(int x, int y)
    {
        _lookupCount++;
        if (_lookupCount % 1000 == 0)
        {
            _stopwatch.Stop();
            Console.WriteLine($"Lookup time for last 1000 lookups: {_stopwatch.ElapsedMilliseconds} ms");
            _stopwatch.Reset();
            _stopwatch.Start();
        }

        int hash = GetHash(x, y);
        foreach (var key in GetNearbyHashes(hash))
        {
            if (_dict.ContainsKey(key))
            {
                foreach (var point in _dict[key])
                {
                    int dx = point.X - x;
                    int dy = point.Y - y;
                    if (dx * dx + dy * dy < _minDistance * _minDistance)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private int GetHash(int x, int y)
    {
        // Using a simple hash function for the sake of example
        return (x / _minDistance) * 31 + (y / _minDistance);
    }

    private IEnumerable<int> GetNearbyHashes(int hash)
    {
        yield return hash;
        // Add more hashes if needed to cover nearby regions
    }
}
