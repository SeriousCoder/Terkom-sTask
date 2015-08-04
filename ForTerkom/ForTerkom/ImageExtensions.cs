using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace ForTerkom
{
    class ImageExtensions
    {
        [DllImport("Normalize.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CalculateHistograme")]
        public static extern void CalculateHistograme(int[,] source, int imgWidth, int imgHeight, int[] hist);

        private Bitmap _image;
        int[] _histR = new int[256];
        int[] _histG = new int[256];
        int[] _histB = new int[256];

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
            var imageR = LibColors(1);
            var imageG = LibColors(2);
            var imageB = LibColors(3);

            CalculateHistograme(imageR, Width, Height, _histR);
            CalculateHistograme(imageG, Width, Height, _histG);
            CalculateHistograme(imageB, Width, Height, _histB);
        }

    }
}
