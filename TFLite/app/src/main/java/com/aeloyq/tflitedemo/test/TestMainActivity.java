package com.aeloyq.tflitedemo.test;

import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.view.View;

import com.aeloyq.tflitedemo.R;

public class TestMainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_testmain);
    }



    public void ButtonTest1OnClick(View v) {
        Intent intent = new Intent(TestMainActivity.this, CameraFloatActivity.class);
        startActivity(intent);
    }

    public void ButtonTest2OnClick(View v) {
        Intent intent = new Intent(TestMainActivity.this, CameraUintActivity.class);
        startActivity(intent);
    }

    public void ButtonTest3OnClick(View v) {
        Intent intent = new Intent(TestMainActivity.this, MinorImplementActivity.class);
        startActivity(intent);
    }

}
