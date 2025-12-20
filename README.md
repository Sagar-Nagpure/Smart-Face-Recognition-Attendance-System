# Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo_Vision-5C3EE8?style=for-the-badge

![License](https://img.shields.io/badge/License-MIT-green intelligent facial recognition-based attendance system that automates student/employee attendance tracking with real-time face detection and verification.

</div>

***

## 📸 Screenshots



✨ Features

- ✅ Real-time face detection and recognition
- ✅ Automated attendance logging with timestamps
- ✅ Web-based Flask interface
- ✅ Multi-user support
- ✅ Attendance reports and statistics
- ✅ Easy to deploy and configure

***

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Backend |
| OpenCV | Face Detection |
| Flask | Web Framework |
| NumPy | Data Processing |

***

## 📋 Prerequisites

- ✅ Python 3.8+
- ✅ pip package manager
- ✅ Webcam/Camera
- ✅ 500MB disk space

***

## 🚀 Installation

```bash
git clone https://github.com/Sagar-Nagpure/Face-Recognition-Attendance-System-.git
cd Face-Recognition-Attendance-System-

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

***

## 📖 Usage

```bash
python app.py
```

Access in browser: http://localhost:5000/

**Workflow:**
1. Register user faces in the system
2. Start real-time face detection
3. System automatically marks attendance
4. View attendance records in dashboard

***

## 📁 Project Structure

```
Face-Recognition-Attendance-System-/
├── app.py                              # Main application
├── requirements.txt                    # Dependencies
├── haarcascade_frontalface_default.xml # Face detector model
├── templates/                          # HTML templates
├── static/                             # CSS, JS, images
├── Attendance/                         # Data storage
└── README.md                           # Documentation
```

***

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Home page |
| `/register` | GET, POST | Register user |
| `/attendance` | GET, POST | Mark attendance |
| `/records` | GET | View records |
| `/api/detect` | POST | Face detection |

***

## 🚀 Performance Tips

- ✅ Use good lighting conditions
- ✅ Register multiple face samples
- ✅ Optimize resolution to 640x480
- ✅ Use GPU acceleration if available

***

## ✅ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check USB connection |
| Poor detection | Improve lighting |
| Recognition errors | Register more samples |
| Port in use | Change Flask port |

***

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open Pull Request

***

## 📝 License

MIT License - See LICENSE file for details

***

## 👨‍💼 Author

**Sagar Nagpure**
- GitHub: [@Sagar-Nagpure](https://github.com/Sagar-Nagpure)
- Repository: [Face-Recognition-Attendance-System](https://github.com/Sagar-Nagpure/Face-Recognition-Attendance-System-)

***

## 🔒 Security

- ✅ Face data stored locally
- ✅ Proper authentication required
- ✅ GDPR compliant
- ✅ Encrypted sensitive data

***

<div align="center">

Made with dedication by Sagar Nagpure

</div>

***
