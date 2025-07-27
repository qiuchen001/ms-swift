# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class CustomPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


class DrivingVideoMultiClassificationPreprocessor(ResponsePreprocessor):
    """
    汽车驾驶视频多分类任务的数据预处理器
    支持一个视频属于多个类别的场景
    """
    
    prompt = """Task: Analyze the driving video and identify ALL applicable driving scenarios from the following categories:
- 白天驾驶 (Day Driving)
- 夜间驾驶 (Night Driving) 
- 城市道路 (Urban Road)
- 高速公路 (Highway)
- 乡村道路 (Rural Road)
- 雨天驾驶 (Rainy Driving)
- 雪天驾驶 (Snowy Driving)
- 拥堵交通 (Traffic Jam)
- 停车场景 (Parking)
- 其他 (Other)

A video can belong to multiple categories. For example, a video can be "白天驾驶 + 城市道路 + 拥堵交通" if it shows daytime driving on urban roads with traffic congestion.

Please provide your analysis in the following format:
<think>
[Your detailed reasoning about the video content, analyzing lighting conditions, road type, weather conditions, traffic density, and any other relevant factors that support your classification]
</think>
<answer>
[Comma-separated list of ALL applicable categories, e.g., "白天驾驶,城市道路,拥堵交通"]
</answer>

Video: {video_path}
Analysis: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        预处理视频多分类数据
        
        Args:
            row: 包含视频路径和标签的数据行
                标签可以是字符串（逗号分隔）或列表格式
            
        Returns:
            处理后的数据字典
        """
        # 提取视频路径和标签
        video_path = row.get('video_path', '')
        label = row.get('label', '')
        
        # 处理标签格式（支持字符串和列表）
        if isinstance(label, list):
            # 如果是列表，转换为逗号分隔的字符串
            label_str = ','.join([str(item).strip() for item in label])
        elif isinstance(label, str):
            # 如果是字符串，直接使用
            label_str = label
        else:
            # 其他格式，转换为字符串
            label_str = str(label)
        
        # 构建查询和响应
        query = self.prompt.format(video_path=video_path)
        response = f"<think>\n[分析视频内容，包括光照条件、道路类型、天气状况、交通密度等因素，支持多分类判断]\n</think>\n<answer>\n{label_str}\n</answer>"
        
        return super().preprocess({
            'query': query,
            'response': response,
            'video_path': video_path,  # 保留视频路径供后续使用
            'label': label_str,  # 保留处理后的标签
            'original_label': label  # 保留原始标签格式
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='SetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))

# 注册汽车驾驶视频多分类数据集
register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/driving_video_classification',
        hf_dataset_id='your_dataset/driving_video_classification',  # 需要替换为实际的数据集ID
        preprocess_func=DrivingVideoMultiClassificationPreprocessor(),
    ))

if __name__ == '__main__':
    dataset = load_dataset(['swift/stsb'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')
