# steps/model_trainer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

class CPURandomForestTrainer:
    """CPU异常检测Random Forest训练器"""
    def __init__(self, test_size=0.2, random_state=42):
        """
        初始化训练器
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.training_history = {}
        logger.info(f"🎯 Random Forest训练器初始化")
        logger.info(f"   测试集比例: {test_size}")
        logger.info(f"   随机种子: {random_state}")

    def prepare_training_data(self, feature_df, feature_list, target_col='is_anomaly'):
        """准备训练数据"""
        logger.info("📊 准备训练数据...")
        # 创建特征矩阵
        X = feature_df[feature_list].copy()
        y = feature_df[target_col].copy()
        # 检查数据质量
        logger.info(f"   数据形状: {X.shape}")
        logger.info(f"   异常比例: {y.mean():.3f}")
        logger.info(f"   缺失值: {X.isnull().sum().sum()}")
        # 处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info(f"   训练集: {X_train.shape}, 异常比例: {y_train.mean():.3f}")
        logger.info(f"   测试集: {X_test.shape}, 异常比例: {y_test.mean():.3f}")
        self.training_history.update({
            'feature_names': feature_list,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'anomaly_ratio': y.mean()
        })
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def hyperparameter_tuning(self, X_train, y_train):
        """超参数调优"""
        logger.info("🔍 开始超参数调优...")
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        logger.info(f"   类别权重: {class_weight_dict}")
        # 参数网格
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['auto', 'sqrt']
        }
        # 基础模型
        rf_base = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        # 网格搜索
        grid_search = GridSearchCV(
            rf_base, param_grid,
            cv=3, scoring='f1',
            n_jobs=-1, verbose=1
        )
        logger.info("   执行网格搜索...")
        grid_search.fit(X_train, y_train)
        # 最佳参数
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        logger.info(f"   ✅ 最佳F1分数: {best_score:.4f}")
        logger.info(f"   ✅ 最佳参数: {best_params}")
        self.training_history.update({
            'best_params': best_params,
            'best_cv_score': best_score,
            'class_weights': class_weight_dict
        })
        return grid_search.best_estimator_

    def train_model(self, X_train, y_train):
        """训练最终模型"""
        logger.info("🚀 训练最终模型...")
        # 超参数调优
        self.model = self.hyperparameter_tuning(X_train, y_train)
        # 交叉验证评估
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=3, scoring='f1')
        logger.info(f"   交叉验证F1分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        # 重新训练完整模型
        self.model.fit(X_train, y_train)
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.training_history['feature_names'],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info(f"   ✅ 模型训练完成!")
        logger.info(f"   📊 特征重要性前5: ")
        for idx, row in self.feature_importance.head().iterrows():
            logger.info(f"      {row['feature']}: {row['importance']:.4f}")
        self.training_history.update({
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'feature_importance': self.feature_importance.to_dict('records')
        })
        return self.model

    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        logger.info("📈 评估模型性能...")
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        # 分类报告
        logger.info("\n📋 分类报告:")
        logger.info(classification_report(y_test, y_pred))
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n🔄 混淆矩阵:")
        logger.info(f"真负例: {cm[0,0]}, 假正例: {cm[0,1]}")
        logger.info(f"假负例: {cm[1,0]}, 真正例: {cm[1,1]}")
        # ROC和PR曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        logger.info(f"\n🎯 关键指标:")
        logger.info(f"   ROC AUC: {roc_auc:.4f}")
        logger.info(f"   PR AUC: {pr_auc:.4f}")
        # 保存评估结果
        self.training_history.update({
            'test_roc_auc': roc_auc,
            'test_pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
        })
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm
        }

from typing import Tuple, Dict, List, Any

@step
def train_model_step(
    feature_df: pd.DataFrame,
    feature_list: list,
    target_col: str = 'is_anomaly',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, StandardScaler, List[str], Dict[str, Any]]:
    """
    训练CPU异常检测随机森林模型

    Args:
        feature_df: 包含特征的DataFrame
        feature_list: 特征列表
        target_col: 目标列名
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        model: 训练好的模型
        scaler: 特征标准化器
        feature_names: 特征名称列表
        training_history: 训练历史记录
    """
    logger.info("🚀 开始CPU异常检测模型训练...")

    # 初始化训练器
    trainer = CPURandomForestTrainer(test_size=test_size, random_state=random_state)

    # 准备训练数据
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = trainer.prepare_training_data(
        feature_df, feature_list, target_col=target_col
    )

    # 训练模型
    model = trainer.train_model(X_train_scaled, y_train)

    # 评估模型
    trainer.evaluate_model(X_test_scaled, y_test)

    logger.info("✅ CPU异常检测模型训练完成!")

    # 返回模型、标准化器、特征名称和训练历史
    return model, trainer.scaler, feature_list, trainer.training_history
