import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("جاري قراءة ملف الإكسل... (هذه العملية ستتم لمرة واحدة فقط ولن تحتاجها مستقبلاً)")
# قراءة البيانات
df_7k = pd.read_excel("7000.xlsx")
imp_df = pd.read_csv("best.vv.csv")
feature_cols = imp_df[imp_df['Importance_R2_Drop'] > 0]['Feature'].tolist()

# تجهيز البيانات والمقياس
train_df, _ = train_test_split(df_7k, test_size=0.2, random_state=42)
train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

# حفظ المقياس
joblib.dump(scaler, "my_scaler.pkl")
print("✅ تم حفظ المقياس بنجاح! يمكنك الآن تشغيل واجهة الويب.")