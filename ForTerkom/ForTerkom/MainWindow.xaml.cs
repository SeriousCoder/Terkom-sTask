using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;


namespace ForTerkom
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool _firstPlay;
        private ImageExtensions _imageExt;

        public MainWindow()
        {
            InitializeComponent();

            _imageExt = new ImageExtensions();
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                OpenFileDialog openFile = new OpenFileDialog
                {
                    Filter = "Video files (*.mp4)|*.mp4",
                    FilterIndex = 1,
                    RestoreDirectory = true
                };

                if (openFile.ShowDialog() == true)
                {
                    Screen.Source = new Uri(openFile.FileName);
                    Screen.LoadedBehavior = MediaState.Pause;
                }

                _firstPlay = true;
            }
            catch (Exception)
            {
                MessageBox.Show("Failed load video");
            }

            
        }

        private void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            Screen.LoadedBehavior = MediaState.Manual;
            if (Screen.Source != null)
            {
                if (PlayButton.Content == "Play" || _firstPlay)
                {
                    PlayButton.Content = "Pause";
                    Screen.Play();
                    _firstPlay = false;
                }
                else
                {
                    PlayButton.Content = "Play";
                    Screen.Pause();
                }
               
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            if (Screen.Source != null) Screen.Stop();
        }

        private void CreateImageButton_Click(object sender, RoutedEventArgs e)
        {
            var bmp = new RenderTargetBitmap(Screen.NaturalVideoWidth, Screen.NaturalVideoHeight, 96, 96, PixelFormats.Pbgra32);
            bmp.Render(Screen);

            _imageExt.ToBitmapImage(bmp);

            ImageItem.Source = bmp;
        }

        

        private void NormalizeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _imageExt.MakeHistogramme();
            }
            catch (Exception)
            {
                MessageBox.Show("Fail");
            }

        }

    }
}
