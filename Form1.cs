using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;

namespace EmguCv
{
    public partial class Form1 : Form
    {
        readonly Dictionary<string, Image<Bgr, byte>> IMGDict;
        public Form1()
        {
            InitializeComponent();
            IMGDict = new Dictionary<string, Image<Bgr, byte>>();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void toolStripMenuOpen_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.Filter = "Image Files (*.jpg;*.png;*.bmp;)|*.jpg;*.png;*.bmp;|All Files (*.*)|*.*;";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    var img = new Image<Bgr, byte>(dialog.FileName);
                    pictureBox1.Image = img.ToBitmap();
                    if (IMGDict.ContainsKey("input"))
                    {
                        IMGDict.Remove("input");
                    }
                    IMGDict.Add("input", img);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

    private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void panel2_Paint(object sender, PaintEventArgs e)
        {

        }

        private void Image_Rec_Click(object sender, EventArgs e)
        {
            try
            {
               
                if (!IMGDict.ContainsKey("input"))
                {
                    throw new Exception("Read an image first");
                }

                var img = IMGDict["input"].Clone().SmoothGaussian(3);
                var blob = DnnInvoke.BlobFromImage(img, 1.0, img.Size, swapRB: true);

                var modelpath = @"C:\Users\VIVEK RAJ\Desktop\MASK_RCNN\frozen_inference_graph.pb";
                var configpath = @"C:\Users\VIVEK RAJ\Desktop\MASK_RCNN\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
                var coconames = @"C:\Users\VIVEK RAJ\Desktop\MASK_RCNN\coco-labels-paper.names";

                var net = DnnInvoke.ReadNetFromTensorflow(modelpath, configpath);
                net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                net.SetPreferableTarget(Target.Cpu);

                var classes = File.ReadAllLines(coconames);

                net.SetInput(blob);

                VectorOfMat output = new();
                string[] outnames = new string[] { "detection_out_final", "detection_masks" };
                net.Forward(output, outnames);

                int numdetections = output[0].SizeOfDimension[2];
                int numclasses = output[1].SizeOfDimension[1];

                var bboxes = output[0].GetData();
                var masks = output[1].GetData();

                for (int i = 0; i < numdetections; i++)
                {
                    float score = (float)bboxes.GetValue(0, 0, i, 2);
                    if (score > 0.6)
                    {
                        int classid = Convert.ToInt32(bboxes.GetValue(0, 0, i, 1));
                        int left = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 3) * img.Cols);
                        int top = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 4) * img.Rows);
                        int right = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 5) * img.Cols);
                        int bottom = Convert.ToInt32((float)bboxes.GetValue(0, 0, i, 6) * img.Rows);


                        left = Math.Max(0, Math.Min(left, img.Cols - 1));
                        right = Math.Max(0, Math.Min(right, img.Cols - 1));

                        top = Math.Max(0, Math.Min(top, img.Rows - 1));
                        bottom = Math.Max(0, Math.Min(bottom, img.Rows - 1));

                        Rectangle rectangle = new(left, top, right - left + 1, bottom - top + 1);
                        img.Draw(rectangle, new Bgr(0, 0, 255), 2);
                        var labels = classes[classid];
                        /*CvInvoke.PutText(img, labels, new Point(left, top - 10), FontFace.HersheySimplex,
                            1.0, new MCvScalar(0, 255, 0), 2);*/
                        CvInvoke.PutText(img, labels, new Point(left, top - 10), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.9, new MCvScalar(255, 255, 255), 2);

                    }
                }

                pictureBox1.Image = img.ToBitmap();
            }
            catch(Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void toolStripMenuExit_Click(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
        }
    }
}