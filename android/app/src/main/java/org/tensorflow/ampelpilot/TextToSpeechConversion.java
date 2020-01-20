package org.tensorflow.ampelpilot;

import android.content.Context;

import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.Toast;

import java.util.Locale;


public class TextToSpeechConversion extends AppCompatActivity {

    private TextToSpeech textToSpeech;
    private Context context;


    TextToSpeechConversion(Context context) {

        this.context = context;

        initTTS();
    }

    private void initTTS() {
        textToSpeech = new TextToSpeech(context, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int ttsLang = textToSpeech.setLanguage(Locale.GERMAN);

                    if (ttsLang == TextToSpeech.LANG_MISSING_DATA || ttsLang == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e("TTS", "Die Sprache wird nicht unterstützt!");
                    } else {
                        Log.i("TTS", "Die Sprache wird unterstützt.");
                    }
                    Log.i("TTS", "Initialisierung erfolgreich.");
                } else {
                    Toast.makeText(getApplicationContext(), "TTS Initialisierung nicht erfolgreich!", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }


    public void speakUp(String speech, boolean flush) {
        int speechStatus;
        if (flush) {
            speechStatus = textToSpeech.speak(speech, TextToSpeech.QUEUE_FLUSH, null);
        } else {
            speechStatus = textToSpeech.speak(speech, TextToSpeech.QUEUE_ADD, null);
        }

        if (speechStatus == TextToSpeech.ERROR) {
            Log.e("TTS", "Error bei der Text to Speech Konvertierung!");
        }
    }

    public void releaseResources() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }

    @Override
    public void onPause() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onPause();
   }

    @Override
    public void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onDestroy();
    }
}







