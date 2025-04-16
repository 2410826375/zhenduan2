import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DIAG_CONFIG = {
    "model_path": "./transformer_fault_cnn.zip",  # 模型文件路径为压缩文件
    "label_encoder_path": "./dataset/label_classes.npy",
    "output_dir": "./diagnosis_results"
}


class AcousticDiagnoser:
    def __init__(self, config):
        self.config = config

        # 解压模型文件
        if not os.path.exists(config["model_path"]):
            raise FileNotFoundError(f"❌ 模型压缩文件不存在: {config['model_path']}")

        # 解压模型文件
        self._extract_model(config["model_path"])

        # 加载模型
        self.model = tf.keras.models.load_model(config["model_path"].replace(".zip", ".h5"), compile=False)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        if not os.path.exists(config["label_encoder_path"]):
            raise FileNotFoundError(f"❌ 标签文件不存在: {config['label_encoder_path']}")
        self.labels = np.load(config["label_encoder_path"])
        if len(self.labels) == 0:
            raise ValueError("❌ 标签文件为空，请检查 label_classes.npy 是否正确生成！")

        os.makedirs(config["output_dir"], exist_ok=True)

    def _extract_model(self, zip_file_path):
        """解压缩模型文件"""
        unzip_dir = os.path.dirname(zip_file_path)
        if not os.path.exists(unzip_dir):
            os.makedirs(unzip_dir)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

        # 确保解压后的文件是 .h5 格式的模型
        extracted_model_path = zip_file_path.replace(".zip", ".h5")
        if not os.path.exists(extracted_model_path):
            raise FileNotFoundError(f"❌ 解压后的模型文件不存在: {extracted_model_path}")
        print(f"✅ 模型已解压至: {extracted_model_path}")

    def diagnose(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 文件不存在: {file_path}")

        if file_path.endswith(".xlsx"):
            return self._diagnose_excel(file_path)
        elif file_path.endswith(".npy"):
            return self._diagnose_npy(file_path)
        else:
            raise ValueError("❌ 当前版本仅支持 .xlsx 和 .npy 文件")

    def _diagnose_excel(self, file_path):
        df = pd.read_excel(file_path)
        time = df["Time"].values
        amplitude = df["Amplitude"].values
        fs = 1 / (time[1] - time[0])
        segment_len = int(fs)
        total_segments = int(np.ceil(len(amplitude) / segment_len))

        results = []

        for i in range(total_segments):
            start_idx = i * segment_len
            end_idx = min((i + 1) * segment_len, len(amplitude))
            segment_amp = amplitude[start_idx:end_idx]
            if len(segment_amp) < segment_len:
                segment_amp = np.pad(segment_amp, (0, segment_len - len(segment_amp)), mode='edge')

            mel_spectrogram = self._segment_to_mel(segment_amp, fs)
            input_data = mel_spectrogram[np.newaxis, ..., np.newaxis]

            pred_probs = self.model.predict(input_data, verbose=0)[0]
            pred_class = self.labels[np.argmax(pred_probs)]
            confidence = np.max(pred_probs)

            self._show_prediction(pred_probs, i + 1)
            self._plot_results(mel_spectrogram, pred_probs, file_path, i + 1)
            result = {
                "段编号": i + 1,
                "预测故障类型": pred_class,
                "置信度": confidence
            }
            results.append(result)

        return results

    def _diagnose_npy(self, file_path):
        mel_spectrogram = np.load(file_path)
        if mel_spectrogram.ndim != 2:
            raise ValueError("❌ .npy 文件内容格式错误，应为二维 MEL 频谱图数组")

        input_data = mel_spectrogram[np.newaxis, ..., np.newaxis]
        pred_probs = self.model.predict(input_data, verbose=0)[0]

        pred_class = self.labels[np.argmax(pred_probs)]
        confidence = np.max(pred_probs)

        self._show_prediction(pred_probs)
        self._plot_results(mel_spectrogram, pred_probs, file_path, segment_index="npy")

        return {
            "预测故障类型": pred_class,
            "置信度": confidence
        }

    def _show_prediction(self, pred_probs, segment_index=None):
        if segment_index is not None:
            print(f"\n🔍 第 {segment_index} 段诊断：")
        for label, prob in zip(self.labels, pred_probs):
            print(f"  - {label}: {prob:.2%}")

    def _segment_to_mel(self, signal, fs):
        import pywt
        from sklearn.preprocessing import MinMaxScaler
        from scipy.signal import butter, filtfilt

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def preprocess(signal, fs, lowcut=50, highcut=1000):
            signal = signal - np.mean(signal)
            b, a = butter_bandpass(lowcut, highcut, fs)
            filtered = filtfilt(b, a, signal)
            scaler = MinMaxScaler(feature_range=(0, 1))
            normed = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
            return normed

        def create_mel_filterbank(num_filters, min_freq, max_freq, cwt_freqs):
            def hz_to_mel(f):
                return 2595 * np.log10(1 + f / 700)

            def mel_to_hz(m):
                return 700 * (10 ** (m / 2595) - 1)

            min_mel = hz_to_mel(min_freq)
            max_mel = hz_to_mel(max_freq)
            mel_points = np.linspace(min_mel, max_mel, num_filters + 2)
            hz_points = mel_to_hz(mel_points)
            bin_indices = [np.argmin(np.abs(cwt_freqs - hz)) for hz in hz_points]
            bin_indices = np.array(bin_indices)
            filterbank = np.zeros((num_filters, len(cwt_freqs)))

            for i in range(num_filters):
                left, center, right = bin_indices[i:i + 3]
                if left < center:
                    filterbank[i, left:center] = np.linspace(0, 1, center - left)
                if center < right:
                    filterbank[i, center:right] = np.linspace(1, 0, right - center)
            return filterbank

        wavelet = "morl"
        scales_num = 300
        min_freq = 50
        max_freq = 0.5 * fs
        n_mels = 50

        segment = preprocess(signal, fs)
        target_freqs = np.linspace(min_freq, max_freq, scales_num)
        scales = pywt.frequency2scale(wavelet, target_freqs / fs)
        cwt_coeffs, _ = pywt.cwt(segment, scales, wavelet, 1 / fs)

        cwt_freqs = pywt.scale2frequency(wavelet, scales) * fs
        sort_idx = np.argsort(cwt_freqs)
        cwt_freqs = cwt_freqs[sort_idx]
        cwt_coeffs = cwt_coeffs[sort_idx, :]

        mel_filterbank = create_mel_filterbank(n_mels, min_freq, cwt_freqs[-1], cwt_freqs)
        mel_energy = np.dot(mel_filterbank, np.abs(cwt_coeffs) ** 2)
        mel_db = 10 * np.log10(mel_energy + 1e-12)
        return mel_db

    def _plot_results(self, mel_spectrogram, pred_probs, file_path, segment_index):
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 1, 1)
            plt.barh(self.labels, pred_probs)
            plt.title("故障概率分布")
            plt.tight_layout()
            basename = os.path.splitext(os.path.basename(file_path))[0]
            save_name = f"{basename}_segment{segment_index}.png"
            save_path = os.path.join(self.config["output_dir"], save_name)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"❌ 可视化失败: {str(e)}")


if __name__ == "__main__":
    diagnoser = AcousticDiagnoser(DIAG_CONFIG)
    test_file = "./正常.xlsx"
    if os.path.exists(test_file):
        reports = diagnoser.diagnose(test_file)
        print("\n✅ 全部诊断完成。")
    else:
        print(f"❌ 测试文件不存在: {test_file}")
