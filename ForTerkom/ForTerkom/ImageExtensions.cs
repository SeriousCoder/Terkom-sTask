using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Color = System.Drawing.Color;

namespace ForTerkom
{
    class ImageExtensions
    {
        [DllImport("Normalize.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CalculateHistograme")]
        public static extern void CalculateHistograme(int[,] source, int imgWidth, int imgHeight, int[] hist);
        [DllImport("Normalize.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
        public static extern void Normalize(int[,] source, int imgWidth, int imgHeight, int histMax, int histMin);

        private Bitmap _image;
        int[] _histR = new int[256];
        int[] _histG = new int[256];
        int[] _histB = new int[256];

        private int[,] imageR;
        private int[,] imageG;
        private int[,] imageB;

        int Width
        {
            get { return _image.Width; }
            set { Width = value; } 
        }
        int Height
        {
            get { return _image.Height; }
            set { Height = value; }
        }

        public void ToBitmapImage(RenderTargetBitmap source)
        {
            var stream = new MemoryStream();
            var bitmapEncoder = new BmpBitmapEncoder();

            bitmapEncoder.Frames.Add(BitmapFrame.Create(source));
            bitmapEncoder.Save(stream);

            _image = new Bitmap(stream);

            imageR = new int[Width, Height];
            imageG = new int[Width, Height];
            imageB = new int[Width, Height];

        }

        int[,] LibColors(int color)
        {
            var foo = new int[Width, Height];

            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    switch (color)
                    {
                        case 1:
                            foo[i, j] = _image.GetPixel(i, j).R;
                            break;
                        case 2:
                            foo[i, j] = _image.GetPixel(i, j).G;
                            break;
                        default:
                            foo[i, j] = _image.GetPixel(i, j).B;
                            break;
                    }

                }
            }

            return foo;
        }

        public void MakeHistogramme()
        {
            imageR = LibColors(1);
            imageG = LibColors(2);
            imageB = LibColors(3);

            CalculateHistograme(imageR, Width, Height, _histR);
            CalculateHistograme(imageG, Width, Height, _histG);
            CalculateHistograme(imageB, Width, Height, _histB);
        }

        public void Normalize()
        {
            Normalize(imageR, Width, Height, _histR.Max(), _histR.Min());
            Normalize(imageG, Width, Height, _histG.Max(), _histG.Min());
            Normalize(imageB, Width, Height, _histB.Max(), _histB.Min());

            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    int r = imageR[i, j];
                    int g = imageG[i, j];
                    int b = imageB[i, j];

                    _image.SetPixel(i, j, Color.FromArgb(r, g, b));
                }
            }
        }

        public ImageSource GetImage()
        {
            var stream = new MemoryStream();
            
            _image.Save(stream, System.Drawing.Imaging.ImageFormat.Bmp);

            var newImage = new BitmapImage();
            newImage.BeginInit();
            stream.Seek(0, SeekOrigin.Begin);
            newImage.StreamSource = stream;
            newImage.EndInit();

            return newImage;
        }

    }
}
