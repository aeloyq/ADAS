package com.aeloyq.tflitedemo;

import android.os.Bundle;
import android.app.Activity;
import android.view.View;
import android.content.Intent;

import com.aeloyq.tflitedemo.test.TestMainActivity;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void Button1OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorADASFaceFloatActivity.class);
        startActivity(intent);
    }

    public void Button2OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorADASFaceUintActivity.class);
        startActivity(intent);
    }

    public void Button3OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
        startActivity(intent);
    }

    public void Button4OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, TestMainActivity.class);
        startActivity(intent);
    }
}
