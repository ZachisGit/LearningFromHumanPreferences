using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IconCreator
{
    class Program
    {
        static OverrideMode overrideMode = OverrideMode.Always; //0 = never, 1 = ask, 2 = always


        static string cDir;

        static BinaryWriter bw;
        static PreferScaling preferScaling = PreferScaling.Down;
        static bool autoClose = true;
        static InterpolationMode upscaleInterpolation = InterpolationMode.NearestNeighbor;
        static InterpolationMode downscaleInterpolation = InterpolationMode.Default;

        static bool cursor = false;

        enum PreferScaling
        {
            Up,         // /su
            Down,       // /sd
            Nearest     // /sn
        }
        enum OverrideMode
        {
            Never,      // /s0
            Ask,        // /s1
            Always      // /s2
        }

        static void Main(string[] args)
        {
            int[] sizes = new int[] { 16, 24, 32, 48, 256 };
            string outputPath = "";
            bool includeFileSizes = true;
            byte cursorX = 0;
            byte cursorY = 0;
            bool autoStartCmd = false;

            cDir = Path.GetTempPath() + "IconCreator\\"; //Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName);
            Directory.CreateDirectory(cDir);


            string help = @"options:
(path1) (path2) (path3) ... // source imgs for icon creation 
/d0  // downscale using nearest neighbour
/d1  // downscale using interpolation (std)
/i0  // only use specified sizes
/i1  // include all input files to the icon (std)
/out (path) // icon output path (std: replace path extension with .ico)
/o0  // override never
/o1  // override ask
/o2  // override always (std)
/s (size1) (size2) (size3) ... / // set included output sizes " + "\n"+@"         // (std: '/s 16 24 32 48 256 /')
/sd  // prefer downscaling (std)
/sn  // prefer nearest scaling
/su  // prefer upscaling
/u0  // upscale using nearest neighbour (std)
/u1  // upscale using interpolation
/x0  // after program is finished, wait for <Enter> to close program
/x1  // auto close program when finished (std)
/a   // auto start command (don't ask the user to add commands) only works when input paths are given
/cur (hotspotX) (hotspotY)   // generate .cur file (not std)";

            if (args.Length == 0)
                Console.WriteLine(help);

            List<string> argsList = args.ToList();

            string optionsCmd = "";
            if (File.Exists(Environment.CurrentDirectory + "\\iconoptions.txt"))
            {
                optionsCmd = File.ReadAllText(Environment.CurrentDirectory + "\\iconoptions.txt", Encoding.Default);
                if (optionsCmd != "")
                {
                    optionsCmd += " / "; // " / " for unused command to seperate commands with multiple parameters from input file paths (f.ex. /s 32 / "C:\\path")
                    if (args.Length > 0)
                        Console.WriteLine("\noptions loaded:\n" + optionsCmd);
                    argsList.AddRange(SplitCommandString(optionsCmd));
                }
            }

            autoStartCmd = argsList.Contains("/a");
            
            //if (args.Length == 0)
            if (!autoStartCmd)
            {
                Console.WriteLine("\ntype your command:");

                for (int i = 0; i < args.Length; i++)
                    Console.Write(args[i] + " ");

                if (optionsCmd != "")
                    Console.Write(optionsCmd);

                string cmd = Console.ReadLine();

                argsList.AddRange(SplitCommandString(cmd));
            }

            args = argsList.ToArray();


            argsList = new List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].Length > 0 && args[i][0] == '/')
                {
                    //command
                    switch (args[i])
                    {
                        case "/?":
                            Console.WriteLine(help + "\n\nPress <Enter> to continue");
                            Console.ReadLine();
                            break;

                        case "/o0": overrideMode = OverrideMode.Never; break;
                        case "/o1": overrideMode = OverrideMode.Ask; break;
                        case "/o2": overrideMode = OverrideMode.Always; break;
                        case "/sd": preferScaling = PreferScaling.Down; break; //down scaling
                        case "/su": preferScaling = PreferScaling.Up; break; //up scaling
                        case "/sn": preferScaling = PreferScaling.Nearest; break; //nearest scaling
                        case "/x0": autoClose = false; break;
                        case "/x1": autoClose = true; break;
                        case "/out":
                            if (i + 1 < args.Length)
                                outputPath = args[++i];
                            break;
                        case "/i0": includeFileSizes = false; break;
                        case "/i1": includeFileSizes = true; break;
                        case "/u0": upscaleInterpolation = InterpolationMode.NearestNeighbor; break;
                        case "/u1": upscaleInterpolation = InterpolationMode.Default; break;
                        case "/d0": downscaleInterpolation = InterpolationMode.NearestNeighbor; break;
                        case "/d1": downscaleInterpolation = InterpolationMode.Default; break;
                        case "/s":
                            List<int> sizesList = new List<int>();
                            while (++i < args.Length)
                            {
                                if (args[i].Length > 0 && args[i][0] != '/')
                                {
                                    //no command -> new size
                                    int newSize;
                                    if (int.TryParse(args[i], out newSize))
                                        sizesList.Add(newSize);
                                    else
                                    {
                                        i--;
                                        break;
                                    }

                                }
                                else
                                {
                                    i--;
                                    break;
                                }
                            }
                            sizes = sizesList.ToArray();

                            break;
                        case "/cur":
                            cursor = true;
                            if (args.Length > ++i && args[i][0] != '/')
                            {
                                byte.TryParse(args[i], out cursorX);

                                if (args.Length > ++i && args[i][0] != '/')
                                    byte.TryParse(args[i], out cursorY);
                            }
                                
                            break;
                    }
                }
                else
                {
                    if (args[i].Length > 0)
                    {
                        if (args[i][0] == '"')
                            argsList.Add(args[i].Substring(1, args[i].Length - 2));
                        else
                            argsList.Add(args[i]);
                        Console.WriteLine(argsList[argsList.Count - 1]);
                    }
                }
            }

            args = argsList.ToArray();


            //CreateIcon("\\output.ico", "\\input5.png");//"\\input1.png", "\\input2.png", "\\input3.png", "\\input4.png", "\\input5.png");//, "input2.png", "input3.png");
            //CreateIconScaled256(outputPath, args);
            if (!cursor)
            {
                outputPath = GetOutputPath(args[0], outputPath);
                CreateIconScaled(outputPath, args, sizes, !includeFileSizes);
            }
            else
                CreateCursors(args, cursorX, cursorY);
            //min: 16, 32, 48, 256
            //all: 16, 20, 24, 32, 40, 48, 64, 96, 256

            Directory.Delete(cDir, true);

            if (!autoClose)
            {
                Console.WriteLine("\nIcon created!\n\nPress <Enter> to close program.");
                Console.ReadLine();
            }
        }

        private static string GetOutputPath(string inputPath, string outputPath = "")
        {
            if (outputPath == "")
                outputPath = (inputPath[0] == '\\' ? "" : Path.GetDirectoryName(inputPath)) + "\\" + Path.GetFileNameWithoutExtension(inputPath) + (cursor ? ".cur" : ".ico");

            if (overrideMode != OverrideMode.Always)
            {
                if (File.Exists(outputPath))
                {
                    bool iterate = true;
                    if (overrideMode == OverrideMode.Ask)
                    {
                        Console.WriteLine("\nWarning: " + outputPath + "\nalready exists, do you want to override it? (y/n)");
                        iterate = Console.ReadLine() != "y";
                    }
                    if (iterate)
                    {
                        //iterate index, until not existsing file is found
                        int index = 0;
                        do
                        {
                            index++;
                            outputPath = Path.GetFileNameWithoutExtension(inputPath) + "_" + index.ToString().PadLeft(3, '0') + ".ico";
                        } while (File.Exists(outputPath));
                    }
                }
            }

            return outputPath;
        }

        static List<string> SplitCommandString(string cmd)
        {
            List<string> argsList = new List<string>();
            for (int i = 0; i < cmd.Length; i++)
            {
                if (cmd[i] == ' ')
                {
                    argsList.Add(cmd.Substring(0, i));
                    cmd = cmd.Substring(i + 1);
                    i = -1;
                }
                else if (cmd[i] == '"')
                {
                    cmd = cmd.Remove(i--, 1);
                    while (++i < cmd.Length && cmd[i] != '"') ;
                    cmd = cmd.Remove(i--, 1);
                }
            }
            if (cmd != "")
                argsList.Add(cmd);

            return argsList;
        }

        static void CreateIconScaled(string outputPath, string[] inputImgs, int[] sizes = null, bool removeNotInSize = false)
        {
            if (sizes == null)
                sizes = new int[0];
            else
            {
                Array.Sort(sizes);
            }

            MakePath(ref inputImgs);
            MakePath(ref outputPath);

            List<int> inputSizes = new List<int>();

            List<Tuple<string, int>> imgs = new List<Tuple<string, int>>();

            //sort input imgs
            for (int i = 0; i < inputImgs.Length; i++)
            {
                using (Bitmap source = new Bitmap(inputImgs[i]))
                {
                    int w = source.Width;
                    int j;
                    for (j = 0; j < imgs.Count && imgs[j].Item2 < w; j++) ;
                    imgs.Insert(j, new Tuple<string, int>(inputImgs[i], w));
                }
            }

            List<Tuple<string, int>> scaled = new List<Tuple<string, int>>();
            for (int i = 0; i < sizes.Length; i++)
            {
                int j;
                for (j = 0; j < imgs.Count; j++)
                {
                    if (imgs[j].Item2 == sizes[i])
                    {
                        j = -1;
                        break;
                    }
                    if (imgs[j].Item2 > sizes[i])
                        break;
                }

                if (j >= 0)
                {
                    InterpolationMode interpolation;

                    if (j == imgs.Count)
                    {
                        //upscaling
                        interpolation = upscaleInterpolation;
                        j = imgs.Count - 1;
                    }
                    else //downscaling
                    {
                        if (j == 0 
                            || 
                            (
                                preferScaling == PreferScaling.Down 
                                ||
                                (preferScaling == PreferScaling.Nearest && (float)imgs[j].Item2 / sizes[i] <= (float)sizes[i] / imgs[j - 1].Item2)) //choose for smallest scale factor
                            )
                        {
                            //downscaling
                            interpolation = downscaleInterpolation;
                        }
                        else
                        {
                            //upscaling
                            j--;
                            interpolation = upscaleInterpolation;
                        }
                    }


                    string scaledPath = cDir + "\\temp" + sizes[i] + Path.GetExtension(imgs[j].Item1);

                    using (Bitmap source = new Bitmap(imgs[j].Item1))
                    using (Bitmap scaledBmp = ResizeBitmap(source, sizes[i], sizes[i], interpolation))
                    {
                        scaledBmp.Save(scaledPath);
                    }

                    scaled.Add(new Tuple<string, int>(scaledPath, sizes[i]));
                }
            }

            if (removeNotInSize)
            {
                //remove imgs, that are not in size
                for (int i = imgs.Count - 1; i >= 0; i--)
                {
                    if (!sizes.Contains(imgs[i].Item2))
                        imgs.RemoveAt(i);
                }
            }
            
            for (int i = 0; i < imgs.Count && scaled.Count > 0; i++)
            {
                if (scaled[0].Item2 < imgs[i].Item2)
                {
                    imgs.Insert(i, scaled[0]);
                    scaled.RemoveAt(0);
                }
            }
            if (scaled.Count > 0)
                imgs.AddRange(scaled);
            

            //remove too large imgs
            for (int i = imgs.Count - 1; i >= 0 && imgs[i].Item2 > 256; i--)
            {
                imgs.RemoveAt(i);
            }

            CreateIcon(outputPath, imgs.Select(f => f.Item1).ToArray());
        }
        
        static Bitmap ResizeBitmap(Bitmap b, int nWidth, int nHeight, InterpolationMode interpolationMode)
        {
            Bitmap result = new Bitmap(nWidth, nHeight);
            using (Graphics g = Graphics.FromImage((Image)result))
            {
                g.InterpolationMode = interpolationMode;
                g.PixelOffsetMode = PixelOffsetMode.Half;
                g.DrawImage(b, 0, 0, nWidth, nHeight);
            }
            return result;
        }

        static void CreateIcon(string outputPath, params string[] inputImgs)
        {
            MakePath(ref outputPath);

            using (FileStream fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            using (bw = new BinaryWriter(fs))
            {
                uint[] imgSizes = new uint[inputImgs.Length];
                for (int i = 0; i < inputImgs.Length; i++)
                {
                    MakePath(ref inputImgs[i]);
                    imgSizes[i] = (uint)(new FileInfo(inputImgs[i])).Length;
                }

                byte type = 1; //1=icon, 2=cursor
                byte imageCount = (byte)inputImgs.Length;

                //Header
                Write(0, 0, type, 0, imageCount, 0); //6b

                uint dataOffset = 6 + 16 * (uint)imageCount;

                //Directory
                for (int i = 0; i < inputImgs.Length; i++)
                {

                    byte width, height, bitsPerPixel;
                    using (Bitmap bmp = new Bitmap(inputImgs[i]))
                    {
                        width = (byte)bmp.Width;
                        height = (byte)bmp.Height;
                        bitsPerPixel = BitsPerPixel(bmp.PixelFormat);
                    }

                    Write(width); //width
                    Write(height); //height
                    Write(0); //color count
                    Write(0);
                    Write(1, 0);//Color planes
                    Write(bitsPerPixel, 0); //Bits per pixel
                    Write(BitConverter.GetBytes(imgSizes[i])); //bitmap size
                    Write(BitConverter.GetBytes(dataOffset)); //bitmap offset in file

                    dataOffset += imgSizes[i];
                }

                //write img data
                for (int i = 0; i < inputImgs.Length; i++)
                {
                    Write(File.ReadAllBytes(inputImgs[i]));
                }
            }
        }

        static void CreateCursors(string[] inputImgs, byte hotspotX, byte hotspotY)
        {
            for (int i = 0; i < inputImgs.Length; i++)
                CreateCursor(GetOutputPath(inputImgs[i]), inputImgs[i], hotspotX, hotspotY);
        }

        static void CreateCursor(string outputPath, string inputImg, byte hotspotX, byte hotspotY)
        {
            MakePath(ref inputImg);
            MakePath(ref outputPath);

            using (FileStream fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            using (bw = new BinaryWriter(fs))
            {
                uint imgSize = (uint)new FileInfo(inputImg).Length;

                byte type = 2; //1=icon, 2=cursor
                byte imageCount = 1;

                //Header
                Write(0, 0, type, 0, imageCount, 0); //6b

                uint dataOffset = 6 + 16 * (uint)imageCount;

                //Directory
                byte width, height, bitsPerPixel;
                using (Bitmap bmp = new Bitmap(inputImg))
                {
                    width = (byte)bmp.Width;
                    height = (byte)bmp.Height;
                    bitsPerPixel = BitsPerPixel(bmp.PixelFormat);
                }

                Write(width); //width
                Write(height); //height
                Write(0); //color count
                Write(0); //reserved
                Write(hotspotX, 0);//x
                Write(hotspotY, 0); //y
                Write(BitConverter.GetBytes(imgSize)); //bitmap size
                Write(BitConverter.GetBytes(dataOffset)); //bitmap offset in file

                dataOffset += imgSize;

                //write img data
                Write(File.ReadAllBytes(inputImg));
            }
        }

        static string MakePath(string myPath)
        {
            if (myPath[0] == '\\')
                return cDir + myPath;
            return myPath;
        }
        static void MakePath(ref string myPath)
        {
            if (myPath[0] == '\\')
                myPath = cDir + myPath;
        }
        static void MakePath(ref string[] myPaths)
        {
            for (int i = 0; i < myPaths.Length; i++)
            {
                MakePath(ref myPaths[i]);
            }
        }
        static void MakePath(List<string> myPaths)
        {
            for (int i = 0; i < myPaths.Count; i++)
            {
                myPaths[i] = MakePath(myPaths[i]);
            }
        }

        //public static byte[] ImageToBytes(Image img)
        //{
        //    using (var stream = new MemoryStream())
        //    {
        //        img.Save(stream, img.RawFormat);
        //        return stream.ToArray();
        //    }
        //}
        static void Write(params byte[] data)
        {
            bw.Write(data);
        }

        static byte BitsPerPixel(System.Drawing.Imaging.PixelFormat format)
        {
            switch (format)
            {
                case System.Drawing.Imaging.PixelFormat.Format1bppIndexed:
                    return 1;

                case System.Drawing.Imaging.PixelFormat.Format4bppIndexed:
                    return 4;

                case System.Drawing.Imaging.PixelFormat.Format8bppIndexed:
                    return 8;

                case System.Drawing.Imaging.PixelFormat.Format16bppGrayScale:
                case System.Drawing.Imaging.PixelFormat.Format16bppRgb555:
                case System.Drawing.Imaging.PixelFormat.Format16bppRgb565:
                case System.Drawing.Imaging.PixelFormat.Format16bppArgb1555:
                    return 16;

                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                    return 24;

                case System.Drawing.Imaging.PixelFormat.Format32bppRgb:
                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                case System.Drawing.Imaging.PixelFormat.Format32bppPArgb:
                    return 32;

                case System.Drawing.Imaging.PixelFormat.Format48bppRgb:
                    return 48;

                case System.Drawing.Imaging.PixelFormat.Format64bppArgb:
                case System.Drawing.Imaging.PixelFormat.Format64bppPArgb:
                    return 64;

                default:
                    return 0;
            }
        }
    }
}
