import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
import warnings
from sklearn.model_selection import train_test_split
# 忽略警告
warnings.filterwarnings('ignore')


model_path = r"model/rf_best_model.pkl"
rb_model = joblib.load(model_path)

if isinstance(rb_model, Pipeline):
    print("Pipeline steps:", rb_model.named_steps)
    rb_estimator = rb_model.named_steps['rf']
else:
    rb_estimator = rb_model
    print("Model is not a Pipeline")

# 加载训练集数据
import pandas as pd
df = pd.read_excel("train1.xlsx")
# 划分特征和目标变量
X = df.drop(['Group'], axis=1)
y = df['Group']
# 划分训练集和测试集
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                     random_state=43, stratify=df['Group'])

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(rb_estimator)

# 计算 SHAP 值
shap_values = explainer.shap_values(x_train)

# 确保输出目录存在
output_dir = "shap_dependence_plots"
os.makedirs(output_dir, exist_ok=True)

# 获取特征名称
if hasattr(x_train, 'columns'):
    feature_names = x_train.columns.tolist()
else:
    # 如果x_train是numpy数组，创建默认特征名
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]

# 检查shap_values的形状
if isinstance(shap_values, list):
    # 多分类情况
    num_classes = len(shap_values)
    print(f"Model has {num_classes} classes")
    
    # 为每个类别和每个特征创建依赖图
    for class_idx in range(num_classes):
        class_output_dir = os.path.join(output_dir, f"class_{class_idx}")
        os.makedirs(class_output_dir, exist_ok=True)
        
        # 为每个特征创建单独的依赖图
        for i, feature_name in enumerate(feature_names):
            plt.figure(figsize=(5, 5))
            
            # 创建简单的散点图，只显示特征值和对应的SHAP值
            feature_values = x_train.iloc[:, i] if hasattr(x_train, 'iloc') else x_train[:, i]
            feature_shap_values = shap_values[class_idx][:, i]
            
            plt.scatter(feature_values, feature_shap_values, alpha=0.5)
            plt.xlabel(feature_name)
            plt.ylabel(f'SHAP value (impact on model output)')
            plt.title(f"SHAP Dependence Plot for {feature_name} (Class {class_idx})")
            
            # 添加水平参考线表示零影响
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            #plt.plot(sorted(feature_values), p(sorted(feature_values)), "r--", alpha=0.5)
            
            plt.tight_layout()
            
            # 保存为PDF
            output_file = os.path.join(class_output_dir, f"{feature_name}_dependence_class_{class_idx}.pdf")
            plt.savefig(output_file)
            
            # 也保存为PNG以便快速预览
            output_file_png = os.path.join(class_output_dir, f"{feature_name}_dependence_class_{class_idx}.png")
            plt.savefig(output_file_png, dpi=150)
            
            plt.close()
            
            print(f"Saved dependence plot for {feature_name} (Class {class_idx})")
else:
    # 二分类或回归情况
    # 为每个特征创建单独的依赖图
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(5, 5))
        
        # 创建简单的散点图，只显示特征值和对应的SHAP值
        feature_values = x_train.iloc[:, i] if hasattr(x_train, 'iloc') else x_train[:, i]
        feature_shap_values = shap_values[:, i]
        
        plt.scatter(feature_values, feature_shap_values, alpha=0.7)
        plt.xlabel(feature_name)
        plt.ylabel(f'SHAP value (impact on model output)')
        plt.title(f"SHAP Dependence for {feature_name}")
        
        # 添加水平参考线表示零影响
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # 添加趋势线
        #z = np.polyfit(feature_values, feature_shap_values, 1)
        #p = np.poly1d(z)
        #plt.plot(sorted(feature_values), p(sorted(feature_values)), "r--", alpha=0.5)
        
        plt.tight_layout()
        
        # 保存为PDF
        output_file = os.path.join(output_dir, f"{feature_name}_dependence.pdf")
        plt.savefig(output_file)
        
        # 也保存为PNG以便快速预览
        output_file_png = os.path.join(output_dir, f"{feature_name}_dependence.png")
        plt.savefig(output_file_png, dpi=150)
        
        plt.close()
        
        print(f"Saved dependence plot for {feature_name}")

print("All SHAP dependence plots have been generated successfully.")