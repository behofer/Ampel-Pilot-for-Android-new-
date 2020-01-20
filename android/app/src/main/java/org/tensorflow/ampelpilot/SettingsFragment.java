package org.tensorflow.ampelpilot;

import android.os.Bundle;
import android.support.v7.preference.PreferenceFragmentCompat;

import org.tensorflow.ampelpilot.R;

public class SettingsFragment extends PreferenceFragmentCompat {

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        setPreferencesFromResource(R.xml.settings_pref, rootKey);
    }
}
