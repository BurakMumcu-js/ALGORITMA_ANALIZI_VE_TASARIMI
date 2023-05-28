import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class Main {
    public static void main(String[] args) {
        // OpenCV'yi başlat
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Görsellerin bulunduğu dizini belirleyin
        String imageDirectory = "./ALGORTIMA_ANALIZI_ODEV_SOURCES/";

        // Görsel dosyalarını oku ve bir diziye kaydet
        File[] imageFiles = new File(imageDirectory).listFiles();

        // Benzerlik eşiği değerini belirleyin (0.0 - 1.0 arasında)
        double similarityThreshold = 0.9;

        // Tüm görselleri karşılaştırın
        for (int i = 0; i < imageFiles.length; i++) {
            File imageFile1 = imageFiles[i];
            BufferedImage image1 = readImage(imageFile1);

            for (int j = i + 1; j < imageFiles.length; j++) {
                File imageFile2 = imageFiles[j];
                BufferedImage image2 = readImage(imageFile2);

                double similarity = compareImages(image1, image2);
                if (similarity >= similarityThreshold) {
                    System.out.println("Benzerlik Oranı: " + similarity);
                    System.out.println("Görsel 1: " + imageFile1.getName());
                    System.out.println("Görsel 2: " + imageFile2.getName());
                    System.out.println("----------------------");
                }
            }
        }
    }

    private static BufferedImage readImage(File imageFile) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(imageFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    private static double compareImages(BufferedImage image1, BufferedImage image2) {
        Mat mat1 = bufferedImageToMat(image1);
        Mat mat2 = bufferedImageToMat(image2);

        Mat hist1 = calculateHistogram(mat1);
        Mat hist2 = calculateHistogram(mat2);

        return Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
    }

    private static Mat bufferedImageToMat(BufferedImage image) {
        Mat mat = new Mat(image.getHeight(), image.getWidth(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    private static Mat calculateHistogram(Mat image) {
        Mat hist = new Mat();
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        MatOfInt channels = new MatOfInt(0);
        Imgproc.calcHist(
                Arrays.asList(image),
                channels,
                new Mat(),
                hist,
                histSize,
                ranges
        );
        Core.normalize(hist, hist, 0, hist.rows(), Core.NORM_MINMAX, -1, new Mat());
        return hist;
    }
}