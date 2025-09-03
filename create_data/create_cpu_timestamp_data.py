import pandas as pd
import numpy as np
import os


# ============ CPU数据生成核心函数 ============

class CPUDataGenerator:
    """CPU使用率数据生成器 - 专为训练Random Forest异常检测模型设计"""

    def __init__(self, start_date='2024-01-01', end_date='2024-07-01', freq='1min'):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.freq = freq
        self.dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        self.n_points = len(self.dates)

        print(f"📊 数据生成器初始化完成")
        print(f"   时间跨度: {start_date} 到 {end_date}")
        print(f"   数据点数: {self.n_points:,} 个")
        print(f"   天数: {(self.end_date - self.start_date).days} 天")

    def generate_base_patterns(self):
        """生成基础的业务模式"""
        print("🔧 生成基础业务模式...")
        hours = self.dates.hour.values
        days_of_week = self.dates.dayofweek.values
        days_of_month = self.dates.day.values
        months = self.dates.month.values

        daily_pattern = 25 + 35 * np.sin(2 * np.pi * (hours - 6) / 24)
        daily_pattern = np.clip(daily_pattern, 15, 60)
        weekly_pattern = np.where(days_of_week < 5, 15, -8)
        monthly_pattern = 8 * np.sin(2 * np.pi * days_of_month / 30) + 3 * np.sin(4 * np.pi * days_of_month / 30)
        seasonal_trend = 5 * np.sin(2 * np.pi * months / 12) + 0.002 * np.arange(self.n_points)
        noise = np.random.normal(0, 4, self.n_points)

        base_cpu = daily_pattern + weekly_pattern + monthly_pattern + seasonal_trend + noise
        base_cpu = np.clip(base_cpu, 5, 85)

        print(f"   ✅ 基础模式生成完成")
        print(f"   CPU范围: {base_cpu.min():.1f}% - {base_cpu.max():.1f}%")
        print(f"   平均CPU: {base_cpu.mean():.1f}%")

        return base_cpu.copy()

    def inject_spike_anomalies(self, cpu_data, ratio=0.025):
        """注入突发型异常"""
        print(f"⚡ 注入突发型异常 (目标比例: {ratio:.1%})...")
        anomaly_indices = []
        anomaly_count = int(self.n_points * ratio)
        candidate_indices = np.arange(100, self.n_points - 100)
        spike_starts = np.random.choice(candidate_indices, anomaly_count // 3, replace=False)

        for start_idx in spike_starts:
            duration = np.random.randint(3, 16)
            end_idx = min(start_idx + duration, self.n_points)
            spike_intensity = np.random.uniform(40, 60)
            spike_shape = np.concatenate([
                np.linspace(0, 1, duration // 3),
                np.ones(duration // 3),
                np.linspace(1, 0, duration - 2 * (duration // 3))
            ])[:duration]

            for i, idx in enumerate(range(start_idx, end_idx)):
                cpu_data[idx] += spike_intensity * spike_shape[i]
                anomaly_indices.append(idx)

        print(f"   ✅ 突发型异常: {len(anomaly_indices)} 个时点")
        return cpu_data, anomaly_indices

    def inject_sustained_anomalies(self, cpu_data, ratio=0.032):
        """注入持续型异常"""
        print(f"📈 注入持续型异常 (目标比例: {ratio:.1%})...")
        anomaly_indices = []
        anomaly_count = int(self.n_points * ratio)
        num_segments = np.random.randint(4, 7)
        segment_size = anomaly_count // num_segments

        for _ in range(num_segments):
            start_idx = np.random.randint(200, self.n_points - segment_size - 200)
            duration = np.random.randint(segment_size // 2, segment_size * 2)
            end_idx = min(start_idx + duration, self.n_points)
            base_increase = np.random.uniform(25, 45)
            fluctuation = np.random.normal(0, 5, end_idx - start_idx)

            for i, idx in enumerate(range(start_idx, end_idx)):
                cpu_data[idx] += base_increase + fluctuation[i]
                anomaly_indices.append(idx)

        print(f"   ✅ 持续型异常: {len(anomaly_indices)} 个时点")
        return cpu_data, anomaly_indices

    def inject_pattern_anomalies(self, cpu_data, ratio=0.016):
        """注入模式型异常"""
        print(f"🌊 注入模式型异常 (目标比例: {ratio:.1%})...")
        anomaly_indices = []
        hours = self.dates.hour.values

        night_candidates = np.where((hours >= 22) | (hours <= 6))[0]
        if len(night_candidates) > 0:
            night_anomalies = np.random.choice(night_candidates, int(len(night_candidates) * 0.05), replace=False)
            for idx in night_anomalies:
                cpu_data[idx] += np.random.uniform(30, 50)
                anomaly_indices.extend(range(max(0, idx - 2), min(self.n_points, idx + 3)))

        weekend_candidates = np.where(self.dates.dayofweek >= 5)[0]
        if len(weekend_candidates) > 0:
            weekend_anomalies = np.random.choice(weekend_candidates, int(len(weekend_candidates) * 0.03), replace=False)
            for idx in weekend_anomalies:
                cpu_data[idx] += np.random.uniform(25, 40)
                anomaly_indices.extend(range(max(0, idx - 1), min(self.n_points, idx + 2)))

        anomaly_indices = list(set(anomaly_indices))
        print(f"   ✅ 模式型异常: {len(anomaly_indices)} 个时点")
        return cpu_data, anomaly_indices

    def inject_gradual_anomalies(self, cpu_data, ratio=0.008):
        """注入渐变型异常"""
        print(f"📉 注入渐变型异常 (目标比例: {ratio:.1%})...")
        anomaly_indices = []
        anomaly_count = int(self.n_points * ratio)
        num_segments = np.random.randint(2, 4)

        for _ in range(num_segments):
            start_idx = np.random.randint(500, self.n_points - anomaly_count // num_segments - 500)
            duration = np.random.randint(anomaly_count // num_segments // 2, anomaly_count // num_segments * 2)
            end_idx = min(start_idx + duration, self.n_points)
            max_increase = np.random.uniform(20, 35)
            gradient = np.linspace(0, max_increase, end_idx - start_idx)
            noise = np.random.normal(0, 3, end_idx - start_idx)

            for i, idx in enumerate(range(start_idx, end_idx)):
                cpu_data[idx] += gradient[i] + noise[i]
                if gradient[i] > max_increase * 0.3:
                    anomaly_indices.append(idx)

        print(f"   ✅ 渐变型异常: {len(anomaly_indices)} 个时点")
        return cpu_data, anomaly_indices

    def generate_dataset(self, output_file='cpu_anomaly_data.csv', report_file='cpu.txt'):
        """生成完整数据集并保存为CSV，同时生成统计报告txt"""
        print("\n🚀 开始生成完整数据集...")

        # 1. 生成基础模式
        cpu_data = self.generate_base_patterns()
        final_cpu = cpu_data.copy()
        all_anomaly_indices = set()

        # 2. 注入异常并记录各类数量
        _, spike_indices = self.inject_spike_anomalies(final_cpu.copy())
        all_anomaly_indices.update(spike_indices)

        _, sustained_indices = self.inject_sustained_anomalies(final_cpu.copy())
        all_anomaly_indices.update(sustained_indices)

        _, pattern_indices = self.inject_pattern_anomalies(final_cpu.copy())
        all_anomaly_indices.update(pattern_indices)

        _, gradual_indices = self.inject_gradual_anomalies(final_cpu.copy())
        all_anomaly_indices.update(gradual_indices)

        # 3. 实际注入异常
        final_cpu, _ = self.inject_spike_anomalies(final_cpu)
        final_cpu, _ = self.inject_sustained_anomalies(final_cpu)
        final_cpu, _ = self.inject_pattern_anomalies(final_cpu)
        final_cpu, _ = self.inject_gradual_anomalies(final_cpu)
        final_cpu = np.clip(final_cpu, 0, 100)

        # 4. 生成标签 (1=正常, 0=异常)
        is_anomaly = np.ones(self.n_points)
        is_anomaly[list(all_anomaly_indices)] = 0

        # 5. 转换为 Unix 时间戳（秒）
        unix_timestamps = self.dates.astype(int) // 1_000_000_000

        # 6. 构建 DataFrame
        df = pd.DataFrame({
            'timestamp': unix_timestamps,
            'cpu_utilization': np.round(final_cpu, 2),
            'is_anomaly': is_anomaly.astype(int)
        })

        # 7. 保存为 CSV
        df.to_csv(output_file, index=False)
        csv_path = os.path.abspath(output_file)
        print(f"\n✅ 数据集已保存至: {csv_path}")

        # 8. 生成统计报告写入 cpu.txt
        total_points = len(df)
        normal_count = int(is_anomaly.sum())
        anomaly_count = total_points - normal_count
        normal_ratio = normal_count / total_points
        anomaly_ratio = anomaly_count / total_points

        report_content = f"""CPU 数据集生成报告
==============================

📅 时间范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}
⏰ 采样频率: {self.freq}
📊 总数据点数: {total_points:,}

🔢 异常类型统计:
  - 突发型异常: {len(spike_indices):,} 个时点 ({len(spike_indices) / total_points:.4f})
  - 持续型异常: {len(sustained_indices):,} 个时点 ({len(sustained_indices) / total_points:.4f})
  - 模式型异常: {len(pattern_indices):,} 个时点 ({len(pattern_indices) / total_points:.4f})
  - 渐变型异常: {len(gradual_indices):,} 个时点 ({len(gradual_indices) / total_points:.4f})

✅ 正常数据: {normal_count:,} 个 ({normal_ratio:.4f})
❌ 异常数据: {anomaly_count:,} 个 ({anomaly_ratio:.4f})

📈 CPU 使用率统计:
  - 均值: {final_cpu.mean():.2f}%
  - 最小值: {final_cpu.min():.2f}%
  - 最大值: {final_cpu.max():.2f}%
  - 标准差: {final_cpu.std():.2f}%

📁 CSV 文件: {os.path.basename(output_file)}
🕒 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        txt_path = os.path.abspath(report_file)
        print(f"📄 统计报告已保存至: {txt_path}")

        # 9. 打印预览
        print(f"\n📋 数据预览:")
        print(df.head(10))

        return df


# ============ 执行生成 ============

if __name__ == "__main__":
    generator = CPUDataGenerator(start_date='2024-01-01', end_date='2024-07-01', freq='1min')
    df = generator.generate_dataset('../data/cpu_data_timestamp_test.csv', '../data/cpu_data_timestamp_test.txt')