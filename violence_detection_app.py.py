import cv2
import torch
import torchvision
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame
from torchvision import transforms
import sys
import winsound
import os
import datetime
from PIL import Image, ImageTk

# لضبط الترميز على UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# تحديد مسار النموذج المدرب
MODEL_PATH = r"C:\Users\ckw10\OneDrive\Desktop\ResNet50_BiLSTM_.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل النموذج
print("Model yükleniyor...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
    print("Model başarıyla yüklendi!")
except Exception as e:
    print("Model yüklenemedi:", e)
    sys.exit(1)

# تعريف بنية النموذج: ResNet50 + BiLSTM
class CNN_LSTM(torch.nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=2, num_layers=2):
        super(CNN_LSTM, self).__init__()
        base_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = torch.nn.Sequential(*list(base_model.children())[:-1])  # إزالة الطبقة الأخيرة
        self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers,
                                  batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        features = [self.cnn(x[:, t, :, :, :]).view(batch_size, -1).unsqueeze(1) for t in range(seq_len)]
        features = torch.cat(features, dim=1)  # دمج الميزات من جميع الإطارات
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]  # أخذ ناتج الإطار الأخير
        logits = self.fc(last_out)
        return logits

# تحميل النموذج في وضع التقييم
model = CNN_LSTM(hidden_dim=hidden_dim, num_classes=2, num_layers=2).to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# تحويل الصور إلى تنسيق مناسب للنموذج
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# تحديد مجلد حفظ الصور المكتشفة
save_dir = r"C:\Users\ckw10\OneDrive\Desktop\tasrim 2"
os.makedirs(save_dir, exist_ok=True)

# عرض الصور المحفوظة في الواجهة
def load_saved_images():
    for widget in frame_inside_canvas.winfo_children():
        widget.destroy()

    images = [img for img in os.listdir(save_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    row = 0
    col = 0

    for img_file in sorted(images):
        img_path = os.path.join(save_dir, img_file)
        img = Image.open(img_path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        img_label = tk.Label(frame_inside_canvas, image=img_tk, cursor="hand2", bd=2, relief="groove")
        img_label.image = img_tk
        img_label.grid(row=row, column=col, padx=5, pady=5)

        def show_full_image(path=img_path):
            top = tk.Toplevel(root)
            top.title(f"{os.path.basename(path)}")
            img_full = Image.open(path)
            img_full_tk = ImageTk.PhotoImage(img_full.resize((800, 600)))
            lbl = tk.Label(top, image=img_full_tk)
            lbl.image = img_full_tk
            lbl.pack()
            tk.Label(top, text=os.path.basename(path), font=("Arial", 12)).pack(pady=10)

        img_label.bind("<Button-1>", lambda e, path=img_path: show_full_image(path))

        lbl_text = tk.Label(frame_inside_canvas, text=img_file.split('.')[0], font=("Arial", 8))
        lbl_text.grid(row=row+1, column=col)

        def delete_image(path=img_path):
            try:
                os.remove(path)
                messagebox.showinfo("تم الحذف", f"تم حذف الصورة:\n{os.path.basename(path)}")
                load_saved_images()
            except Exception as e:
                messagebox.showerror("خطأ", f"لم يتم حذف الصورة:\n{str(e)}")

        del_btn = tk.Button(frame_inside_canvas, text="🗑️ ", font=("Arial", 9), fg="white", bg="red",
                            command=lambda p=img_path: delete_image(p))
        del_btn.grid(row=row+2, column=col, pady=(0, 15))

        col += 1
        if col >= 4:
            col = 0
            row += 3

# تحليل الفيديو أو الكاميرا واكتشاف العنف
def analyze_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Video açılamadı!")
        return

    frame_buffer = []
    frame_index = 0
    frame_step = 2
    violence_prev_above_threshold = False  # لتفادي تكرار الحفظ

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        if len(frame_buffer) > 16:
            frame_buffer.pop(0)

        violence_prob = 0.0

        if len(frame_buffer) == 16:
            frames_tensor = torch.stack([
                transform(cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB))
                for f in frame_buffer
            ]).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(frames_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                violence_prob = probs[0][1].item()

            if violence_prob > 0.5 and not violence_prev_above_threshold:
                violence_prev_above_threshold = True

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(save_dir, f"violence_frame_{filename_time}.jpg")

                annotated_frame = frame.copy()
                cv2.putText(
                    annotated_frame,
                    f"Detected: {now}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )

                cv2.imwrite(filename, annotated_frame)
                winsound.Beep(1000, 500)
                load_saved_images()

            if violence_prob <= 0.5:
                violence_prev_above_threshold = False

        bar_height = 50
        frame_height, frame_width = frame.shape[:2]
        bar_color = (0, 255, 0) if violence_prob <= 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (frame_width, bar_height), bar_color, -1)
        text = f"Violence Probability: {violence_prob * 100:.2f}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (bar_height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Şiddet Tespit Sistemi", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Tamamlandı", "Analiz tamamlandı!")

# دوال الأزرار
def select_video(): analyze_video(filedialog.askopenfilename(filetypes=[("Video Dosyaları", "*.mp4 *.avi")]))

def start_camera(): analyze_video(0)

def close_app(): root.destroy()

# إعداد الواجهة الرسومية
root = tk.Tk()
root.title("Violence Detection Using AI (ResNet50 + BiLSTM)")
root.geometry("1000x650")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Violence Detection Using AI (ResNet50 + BiLSTM)", font=("Arial", 18, "bold"),
         bg="#f0f0f0", fg="#333").pack(pady=10)

buttons_frame = tk.Frame(root, bg="#f0f0f0")
buttons_frame.pack()

tk.Button(buttons_frame, text="📂 Analyze Video", font=("Arial", 12), bg="#4CAF50", fg="white",
          command=select_video).grid(row=0, column=0, padx=10, pady=10)
tk.Button(buttons_frame, text="🎥 Open Camera", font=("Arial", 12), bg="#2196F3", fg="white",
          command=start_camera).grid(row=0, column=1, padx=10)
tk.Button(buttons_frame, text="❌ Exit", font=("Arial", 12), bg="#f44336", fg="white",
          command=close_app).grid(row=0, column=2, padx=10)

canvas = Canvas(root, bg="#f0f0f0", height=400)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas, bg="#f0f0f0")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
frame_inside_canvas = scrollable_frame

load_saved_images()
root.mainloop()
