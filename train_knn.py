import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def setup_directories():
    """Create project directories if they do not exist."""
    dirs = [
        'data',
        'static/css',
        'static/js',
        'static/images/analysis',
        'static/models',
        'templates'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Directory verified: {d}")

def generate_synthetic_dataset(num_samples_per_class=50):
    """
    Generate a realistic synthetic dataset representing rice leaf disease visual features.
    
    Features:
    1. Greenness: Ratio of green area (0.0 to 1.0)
    2. BrownSpotDensity: Ratio of brown spots area (0.0 to 1.0)
    3. BlackSpotDensity: Ratio of small black spots/smut area (0.0 to 1.0)
    4. LesionArea: Percentage of yellow/pale lesions (0.0 to 1.0)
    
    Labels:
    - 0: Hawar Daun Bakteri (Bacterial Leaf Blight) - High LesionArea, Low Greenness
    - 1: Bercak Cokelat (Brown Spot) - High BrownSpotDensity, Mid Greenness
    - 2: Smut Daun (Leaf Smut) - High BlackSpotDensity, Mid-High Greenness
    - 3: Sehat (Healthy) - High Greenness, Very Low Spot/Lesion Densities
    """
    np.random.seed(42)
    
    data = []
    
    # Class 0: Hawar Daun Bakteri (Bacterial Leaf Blight)
    for _ in range(num_samples_per_class):
        greenness = np.random.normal(0.35, 0.08)
        brown_spot = np.random.normal(0.08, 0.04)
        black_spot = np.random.normal(0.04, 0.02)
        lesion_area = np.random.normal(0.70, 0.10)
        
        # Clip values to [0.0, 1.0]
        data.append([
            np.clip(greenness, 0.1, 0.6),
            np.clip(brown_spot, 0.0, 0.2),
            np.clip(black_spot, 0.0, 0.1),
            np.clip(lesion_area, 0.4, 0.95),
            0,
            "Hawar Daun Bakteri"
        ])
        
    # Class 1: Bercak Cokelat (Brown Spot)
    for _ in range(num_samples_per_class):
        greenness = np.random.normal(0.55, 0.07)
        brown_spot = np.random.normal(0.65, 0.10)
        black_spot = np.random.normal(0.05, 0.03)
        lesion_area = np.random.normal(0.08, 0.04)
        
        data.append([
            np.clip(greenness, 0.3, 0.75),
            np.clip(brown_spot, 0.4, 0.95),
            np.clip(black_spot, 0.0, 0.15),
            np.clip(lesion_area, 0.0, 0.2),
            1,
            "Bercak Cokelat"
        ])
        
    # Class 2: Smut Daun (Leaf Smut)
    for _ in range(num_samples_per_class):
        greenness = np.random.normal(0.65, 0.06)
        brown_spot = np.random.normal(0.06, 0.03)
        black_spot = np.random.normal(0.60, 0.12)
        lesion_area = np.random.normal(0.05, 0.03)
        
        data.append([
            np.clip(greenness, 0.45, 0.8),
            np.clip(brown_spot, 0.0, 0.15),
            np.clip(black_spot, 0.35, 0.9),
            np.clip(lesion_area, 0.0, 0.15),
            2,
            "Smut Daun"
        ])
        
    # Class 3: Sehat (Healthy)
    for _ in range(num_samples_per_class):
        greenness = np.random.normal(0.88, 0.04)
        brown_spot = np.random.normal(0.02, 0.01)
        black_spot = np.random.normal(0.01, 0.01)
        lesion_area = np.random.normal(0.02, 0.01)
        
        data.append([
            np.clip(greenness, 0.75, 0.98),
            np.clip(brown_spot, 0.0, 0.05),
            np.clip(black_spot, 0.0, 0.04),
            np.clip(lesion_area, 0.0, 0.05),
            3,
            "Sehat"
        ])
        
    columns = ['Greenness', 'BrownSpotDensity', 'BlackSpotDensity', 'LesionArea', 'Label', 'DiseaseName']
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    csv_path = 'data/rice_leaf_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"Synthetic dataset generated successfully at: {csv_path}")
    return df

def generate_visualizations(df):
    """Generate and save Seaborn and Matplotlib visualizations for the dataset."""
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # Palette definition for classes
    colors = {
        "Hawar Daun Bakteri": "#e74c3c",  # Red
        "Bercak Cokelat": "#d35400",       # Orange/Brown
        "Smut Daun": "#2c3e50",            # Dark Slate/Black
        "Sehat": "#2ecc71"                 # Green
    }
    
    # 1. Feature Distribution (Boxplot)
    plt.figure(figsize=(12, 8))
    features = ['Greenness', 'BrownSpotDensity', 'BlackSpotDensity', 'LesionArea']
    df_melted = pd.melt(df, id_vars=['DiseaseName'], value_vars=features, 
                        var_name='Feature', value_name='Value')
    
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='DiseaseName', palette=colors)
    plt.title('Distribusi Ciri Visual Daun Padi berdasarkan Kelas Penyakit', pad=15)
    plt.ylabel('Nilai Fitur (Terskala 0-1)')
    plt.xlabel('Fitur Visual')
    plt.legend(title='Kelas/Kondisi Daun', frameon=True)
    plt.tight_layout()
    plt.savefig('static/images/analysis/boxplot_features.png', dpi=150)
    plt.close()
    print("Visualisasi 1 (Boxplot) berhasil disimpan.")
    
    # 2. Pairplot
    pairplot = sns.pairplot(df.drop(columns=['Label']), hue='DiseaseName', palette=colors, 
                            vars=features, height=2.5, aspect=1.1, diag_kind='kde')
    pairplot.fig.suptitle('Matriks Scatter Plot Ciri Visual Daun Padi', y=1.02)
    pairplot.savefig('static/images/analysis/pairplot_features.png', dpi=150)
    plt.close()
    print("Visualisasi 2 (Pairplot) berhasil disimpan.")
    
    # 3. Heatmap Korelasi Fitur
    plt.figure(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Matriks Korelasi Antar Fitur Visual', pad=15)
    plt.tight_layout()
    plt.savefig('static/images/analysis/correlation_matrix.png', dpi=150)
    plt.close()
    print("Visualisasi 3 (Correlation Matrix) berhasil disimpan.")

def train_and_save_model(df):
    """Train KNN Classifier and save model and scaler."""
    X = df[['Greenness', 'BrownSpotDensity', 'BlackSpotDensity', 'LesionArea']].values
    y = df['Label'].values
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    # We will use K=5 as default with euclidean distance
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test_scaled)
    acc = knn.score(X_test_scaled, y_test)
    print(f"\nEvaluasi Model KNN (K={k}, Euclidean):")
    print(f"Akurasi Data Uji: {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=["Hawar Daun Bakteri", "Bercak Cokelat", "Smut Daun", "Sehat"]))
    
    # Save the model and scaler
    model_path = 'static/models/knn_model.pkl'
    scaler_path = 'static/models/scaler.pkl'
    
    # Wait, save also the X_train_scaled and y_train inside the model or reference it, 
    # so we can easily query neighbors on the exact training set from the web!
    # Yes! A custom dictionary is better.
    model_data = {
        'knn_model': knn,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_train_raw': X_train,
        'feature_names': ['Greenness', 'BrownSpotDensity', 'BlackSpotDensity', 'LesionArea']
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved successfully to: {model_path}")
        
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved successfully to: {scaler_path}")
    
    return acc

if __name__ == '__main__':
    print("=== Memulai Setup & Pelatihan Model KNN ===")
    setup_directories()
    df = generate_synthetic_dataset()
    generate_visualizations(df)
    train_and_save_model(df)
    print("=== Setup & Pelatihan Model Selesai ===")
