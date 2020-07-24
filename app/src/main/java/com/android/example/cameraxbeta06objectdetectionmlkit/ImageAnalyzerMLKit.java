package com.android.example.cameraxbeta06objectdetectionmlkit;

import android.app.Activity;
import android.content.Context;
import android.graphics.Rect;
import android.media.Image;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public class ImageAnalyzerMLKit implements ImageAnalysis.Analyzer {

    //Image Analyzer
    private ObjectDetector objectDetector;
    private List<String> labels;

    private String assetModelName = "mobilenet_v1_1.0_224_quant.tflite";
    private String assetLabelName = "labels_mobilenet_quant_v1_224.txt";

    public String result;

    private Context mContext;
    private Activity mActivity;


    private Rect detectedLocation;
    private String detectedLabel;
    private int detectedConfidence;


    public ImageAnalyzerMLKit(Context context, Activity activity){
        mContext = context;
        mActivity = activity;
    }

    public Rect getDetectedLocation() {return detectedLocation;}
    public String getDetectedLabel() {return detectedLabel;}
    public int getDetectedConfidence() {return detectedConfidence;}

    @Override
    @androidx.camera.core.ExperimentalGetImage
    public void analyze(@NonNull ImageProxy imageProxy) {

        prepareObjectDetector();
        prepareLabels();

        Image mediaImage = imageProxy.getImage();
        if (mediaImage != null) {

            InputImage image = InputImage.fromMediaImage(
                    mediaImage,
                    imageProxy.getImageInfo().getRotationDegrees()
            );

            objectDetector
                    .process(image)
                    .addOnFailureListener(e -> imageProxy.close())
                    .addOnSuccessListener(detectedObjects -> {

                        for (DetectedObject detectedObject : detectedObjects) {
                            for (DetectedObject.Label label : detectedObject.getLabels()) {
                                detectedLocation = detectedObject.getBoundingBox();
                                detectedLabel = labels.get(label.getIndex());
                                detectedConfidence = Math.round(label.getConfidence()*100);

                                TextView labelPreview = mActivity.findViewById(R.id.label);
                                labelPreview.setText(detectedLabel + ": " + detectedConfidence + "%");
                            }
                        }
                        imageProxy.close();
                    });
        }
    }

    private void prepareObjectDetector() {
        CustomObjectDetectorOptions options = new CustomObjectDetectorOptions.Builder(loadModel(assetModelName))
                .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
                //.setDetectorMode(CustomObjectDetectorOptions.STREAM_MODE)
                .enableMultipleObjects()
                .enableClassification()
                .setClassificationConfidenceThreshold(0.5f)
                .setMaxPerObjectLabelCount(3)
                .build();
        objectDetector = ObjectDetection.getClient(options);
    }

    private LocalModel loadModel(String assetFileName) {
        return new LocalModel.Builder()
                .setAssetFilePath(assetFileName)
                .build();
    }

    private void prepareLabels() {
        try {
            InputStreamReader reader = new InputStreamReader(mContext.getAssets().open(assetLabelName));
            labels = readLines(reader);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<String> readLines(InputStreamReader reader) {
        BufferedReader bufferedReader = new BufferedReader(reader, 8 * 1024);
        Iterator<String> iterator = new LinesSequence(bufferedReader);

        ArrayList<String> list = new ArrayList<>();

        while (iterator.hasNext()) {
            list.add(iterator.next());
        }

        return list;
    }

    static class LinesSequence implements Iterator<String> {
        private BufferedReader reader;
        private String nextValue;
        private Boolean done = false;

        public LinesSequence(BufferedReader reader) {
            this.reader = reader;
        }

        @Override
        public boolean hasNext() {
            if (nextValue == null && !done) {
                try {
                    nextValue = reader.readLine();
                } catch (IOException e) {
                    e.printStackTrace();
                    nextValue = null;
                }
                if (nextValue == null) done = true;
            }
            return nextValue != null;
        }

        @Override
        public String next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            String answer = nextValue;
            nextValue = null;
            return answer;
        }
    }

}
