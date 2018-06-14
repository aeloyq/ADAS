package com.aeloyq.tfmobile;

import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.view.View;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void Button1OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorADASFaceActivity.class);
        startActivity(intent);
    }

    public void Button2OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorADASFaceFakeQuantizedActivity.class);
        startActivity(intent);
    }

    public void Button3OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorADASFaceQuantizedActivity.class);
        startActivity(intent);
    }

    public void Button4OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
        startActivity(intent);
    }

    public void Button5OnClick(View v) {
        Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
        startActivity(intent);
    }

}
