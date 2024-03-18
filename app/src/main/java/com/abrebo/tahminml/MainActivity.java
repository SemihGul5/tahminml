package com.abrebo.tahminml;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;

import com.abrebo.tahminml.databinding.ActivityMainBinding;
import com.abrebo.tahminml.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.buttonSec.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, 10);
            binding.textView.setText("");
        });

        binding.buttonTahmin.setOnClickListener(view -> {
            if (img == null) {
                // No image selected, handle this case
                return;
            }

            try {
                Model model = Model.newInstance(getApplicationContext());

                // Create inputs for reference.
                TensorImage tensorImage = new TensorImage(DataType.UINT8);
                tensorImage.load(img);

                // Create an ImageProcessor
                ImageProcessor imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                        // Normalize the pixel values to range [0, 1] (remove this line if the model expects uint8 input)
                        //.add(new NormalizeOp(0, 255))
                        .build();

                // Process the image with the ImageProcessor
                tensorImage = imageProcessor.process(tensorImage);

                // Convert the image to TensorBuffer
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
                inputFeature0.loadBuffer(tensorImage.getBuffer());

                // Run model inference and get the output
                Model.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                // Display the output results
                binding.textView.setText(outputFeature0.getFloatArray()[0] + "\n" + outputFeature0.getFloatArray()[1]);

                // Release model resources if no longer used.
                model.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 10 && resultCode == RESULT_OK) {
            if (data != null) {
                Uri selectedImageUri = data.getData();
                try {
                    // Load the selected image into a Bitmap
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);

                    // Display the selected image
                    binding.imageView.setImageBitmap(img);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
