import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, RotateCcw, Image, Video, Camera, Download } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runDetection, runVideoDetection, getModels, BASE_URL } from '../api';

// Detection overlay component for real-time bounding boxes
interface DetectionOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections: any[];
  isActive: boolean;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({ videoRef, detections, isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!isActive || !canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    // Match canvas size to video element's container
    const resizeCanvas = () => {
      const parent = video.parentElement;
      if (parent) {
        const rect = parent.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [isActive, videoRef]);
  
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (detections.length === 0) return;
    
    // Calculate the actual rendered video dimensions (accounting for object-fit: contain)
    const videoRatio = video.videoWidth / video.videoHeight;
    const canvasRatio = canvas.width / canvas.height;
    
    let renderWidth = canvas.width;
    let renderHeight = canvas.height;
    let offsetX = 0;
    let offsetY = 0;
    
    if (videoRatio > canvasRatio) {
      // Video is wider than canvas - black bars on top/bottom
      renderWidth = canvas.width;
      renderHeight = canvas.width / videoRatio;
      offsetY = (canvas.height - renderHeight) / 2;
    } else {
      // Video is taller than canvas - black bars on sides
      renderHeight = canvas.height;
      renderWidth = canvas.height * videoRatio;
      offsetX = (canvas.width - renderWidth) / 2;
    }
    
    // Calculate scale factors based on the actual rendered video size
    const scaleX = renderWidth / video.videoWidth;
    const scaleY = renderHeight / video.videoHeight;
    
    // Draw bounding boxes with offset
    detections.forEach((det) => {
      const bbox = det.bbox;
      if (!bbox || bbox.length < 4) return;
      
      const x = offsetX + (bbox[0] * scaleX);
      const y = offsetY + (bbox[1] * scaleY);
      const width = (bbox[2] - bbox[0]) * scaleX;
      const height = (bbox[3] - bbox[1]) * scaleY;
      
      // Draw box
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);
      
      // Draw label background
      const label = `${det.class_name || 'crack'} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 14px sans-serif';
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;
      
      ctx.fillStyle = '#22c55e';
      ctx.fillRect(x, y - textHeight, textMetrics.width + 10, textHeight);
      
      // Draw label text
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + 5, y - 5);
    });
  }, [detections, videoRef]);
  
  if (!isActive) return null;
  
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-10"
      style={{ width: '100%', height: '100%' }}
    />
  );
};

const Detection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'photo' | 'video' | 'webcam'>('photo');
  
  // Photo detection states
  const [imagePath, setImagePath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // Video detection states
  const [videoPath, setVideoPath] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [sampleInterval, setSampleInterval] = useState<number>(5);
  
  
  // Webcam detection states
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [autoCapture, setAutoCapture] = useState(true);
  const [captureInterval] = useState<number>(200);
  const [, setWebcamResult] = useState<any>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [fps, setFps] = useState<number>(0);
  
  // Common states
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models, setModels] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string>('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const webcamVideoRef = useRef<HTMLVideoElement>(null);
  const autoCaptureRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getModels().then(res => setModels(res.data));
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setImagePath(file.name);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedVideo(file);
      setVideoPath(file.name);
      setVideoPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const openFilePicker = () => {
    if (activeTab === 'photo') {
      fileInputRef.current?.click();
    } else if (activeTab === 'video') {
      videoInputRef.current?.click();
    }
  };

  // Webcam functions
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false 
      });
      setWebcamStream(stream);
      setWebcamActive(true);
      setAutoCapture(true); // Enable real-time detection automatically
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
  };

  // Effect to connect video element to stream
  useEffect(() => {
    if (webcamActive && webcamStream && webcamVideoRef.current) {
      webcamVideoRef.current.srcObject = webcamStream;
    }
  }, [webcamActive, webcamStream]);

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
    }
    setWebcamActive(false);
    if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
    setAutoCapture(false);
    setWebcamResult(null);
  };

  const captureFromWebcam = async () => {
    if (!webcamVideoRef.current || !selectedModel || isCapturing) return;
    
    setIsCapturing(true);
    const video = webcamVideoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setIsCapturing(false);
        return;
      }
      
      const selectedModelData = models.find(m => m.id === selectedModel);
      const file = new File([blob], `webcam_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
      
      const formData = new FormData();
      formData.append('image', file);
      formData.append('model_path', selectedModelData?.path || '');

      
      const startTime = performance.now();
      
      try {
        const res = await runDetection(formData);
        setWebcamResult(res.data);
        
        // Update detections for overlay
        if (res.data?.success && res.data?.result?.success) {
          setDetections(res.data.result.detections || []);
          
          // Calculate FPS
          const endTime = performance.now();
          const inferenceTime = endTime - startTime;
          setFps(Math.round(1000 / inferenceTime));
        }
      } catch (error) {
        console.error('Detection failed:', error);
      } finally {
        setIsCapturing(false);
      }
    }, 'image/jpeg', 0.8);
  };

  // Auto-capture effect
  useEffect(() => {
    if (autoCapture && webcamActive && selectedModel) {
      autoCaptureRef.current = setInterval(() => {
        captureFromWebcam();
      }, captureInterval);
    } else if (autoCaptureRef.current) {
      clearInterval(autoCaptureRef.current);
      autoCaptureRef.current = null;
    }
    
    return () => {
      if (autoCaptureRef.current) {
        clearInterval(autoCaptureRef.current);
        autoCaptureRef.current = null;
      }
    };
  }, [autoCapture, webcamActive, selectedModel, captureInterval]);

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const handlePredict = async () => {
    const selectedModelData = models.find(m => m.id === selectedModel);
    
    if (activeTab === 'photo') {
      if (!selectedFile || !selectedModel) return;

      setLoading(true);
      setProgress('');

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('model_path', selectedModelData?.path || '');


      try {
        const res = await runDetection(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run detection'
        });
      }
      setLoading(false);
    } else if (activeTab === 'video') {
      if (!selectedVideo || !selectedModel) return;

      setLoading(true);
      setProgress('Extracting frames... This may take a while.');

      const formData = new FormData();
      formData.append('video', selectedVideo);
      formData.append('model_path', selectedModelData?.path || '');

      formData.append('sample_interval', sampleInterval.toString());

      try {
        const res = await runVideoDetection(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run video detection'
        });
      }
      setLoading(false);
      setProgress('');
    }
  };

  return (
    <div className="flex gap-6 h-full overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 space-y-6 overflow-auto pr-2 pb-10 custom-scrollbar">
        {/* Tab Navigation */}
        <div 
          className="flex gap-2 pb-4"
          style={{ borderBottom: '1px solid var(--border-primary)' }}
        >
          <button
            onClick={() => {
              setActiveTab('photo');
              stopWebcam();
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all"
            style={
              activeTab === 'photo'
                ? { background: 'var(--accent-primary)', color: 'white' }
                : { color: 'var(--text-muted)', background: 'transparent' }
            }
            onMouseEnter={(e) => {
              if (activeTab !== 'photo') {
                e.currentTarget.style.background = 'var(--bg-hover)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== 'photo') {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = 'var(--text-muted)';
              }
            }}
          >
            <Image size={18} />
            Photo
          </button>
          <button
            onClick={() => {
              setActiveTab('video');
              stopWebcam();
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all"
            style={
              activeTab === 'video'
                ? { background: 'var(--accent-primary)', color: 'white' }
                : { color: 'var(--text-muted)', background: 'transparent' }
            }
            onMouseEnter={(e) => {
              if (activeTab !== 'video') {
                e.currentTarget.style.background = 'var(--bg-hover)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== 'video') {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = 'var(--text-muted)';
              }
            }}
          >
            <Video size={18} />
            Video
          </button>
          <button
            onClick={() => setActiveTab('webcam')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all"
            style={
              activeTab === 'webcam'
                ? { background: 'var(--accent-primary)', color: 'white' }
                : { color: 'var(--text-muted)', background: 'transparent' }
            }
            onMouseEnter={(e) => {
              if (activeTab !== 'webcam') {
                e.currentTarget.style.background = 'var(--bg-hover)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== 'webcam') {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = 'var(--text-muted)';
              }
            }}
          >
            <Camera size={18} />
            Webcam
          </button>
        </div>

        {/* Configuration Panel */}
        {activeTab !== 'webcam' && (
          <Panel title={`${activeTab === 'photo' ? 'Photo' : 'Video'} Detection Configuration`}>
            <div className="flex gap-4 items-start">
              {/* Model Selection */}
              <div className="w-[250px] shrink-0">
                <label className="text-sm mb-2 block h-[22px]" style={{ color: 'var(--text-muted)' }}>Select Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="input-clean w-full"
                >
                  <option value="">Choose a trained model...</option>
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.id}) - {(model.size / (1024 * 1024)).toFixed(1)} MB
                    </option>
                  ))}
                </select>
                {models.length === 0 && (
                  <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                    No trained models available. Train a model first.
                  </p>
                )}
              </div>

              {/* File Upload */}
              <div className="flex-1 min-w-[300px]">
                <label className="text-sm mb-2 block h-[22px]" style={{ color: 'var(--text-muted)' }}>
                  Select {activeTab === 'photo' ? 'Image' : 'Video'}
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={activeTab === 'photo' ? imagePath : videoPath}
                    readOnly
                    placeholder={`Click Browse to select a ${activeTab}...`}
                    className="input-clean flex-1"
                  />
                  <button
                    onClick={openFilePicker}
                    className="btn-clean px-4 py-2"
                    title="Browse files"
                  >
                    Browse
                  </button>
                </div>
              </div>

              {/* Run Detection Button */}
              <div className="shrink-0">
                <label className="text-sm mb-2 block h-[22px] opacity-0" style={{ color: 'var(--text-muted)' }}>Action</label>
                <Button 
                  primary 
                  onClick={handlePredict} 
                  disabled={loading || !(activeTab === 'photo' ? selectedFile : selectedVideo) || !selectedModel}
                >
                  {loading ? (
                    <>
                      <RotateCcw size={18} className="inline mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play size={18} className="inline mr-2" />
                      Run Detection
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Secondary Controls Row */}
            <div className="flex gap-4 mt-4 flex-wrap items-center">
              {activeTab === 'video' && (
                <div style={{ minWidth: '300px', flex: '1 1 300px' }}>
                  <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Frame Sample Interval (lower = more accurate but slower)</label>
                  <input
                    type="range"
                    min="1"
                    max="30"
                    value={sampleInterval}
                    onChange={(e) => setSampleInterval(parseInt(e.target.value))}
                    className="w-full mt-1"
                  />
                  <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                    <span>1 (all frames)</span>
                    <span>{sampleInterval} frame(s)</span>
                    <span>30 (fast)</span>
                  </div>
                </div>
              )}
              {progress && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--accent-primary)' }} />
                  <p className="text-xs font-medium" style={{ color: 'var(--accent-primary)' }}>{progress}</p>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            <input
              ref={videoInputRef}
              type="file"
              accept="video/*"
              onChange={handleVideoSelect}
              className="hidden"
            />
          </Panel>
        )}

        {/* Webcam Configuration Panel */}
        {activeTab === 'webcam' && (
          <Panel title="Webcam Detection Configuration">
            <div className="flex gap-4 items-start">
              {/* Model Selection */}
              <div className="w-[300px] shrink-0">
                <label className="text-sm mb-2 block h-[22px]" style={{ color: 'var(--text-muted)' }}>Select Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="input-clean w-full"
                >
                  <option value="">Choose a trained model...</option>
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.id}) - {(model.size / (1024 * 1024)).toFixed(1)} MB
                    </option>
                  ))}
                </select>
                {models.length === 0 && (
                  <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                    No trained models available.
                  </p>
                )}
              </div>

              {/* Webcam Control */}
              <div className="shrink-0">
                <label className="text-sm mb-2 block h-[22px]" style={{ color: 'var(--text-muted)' }}>Webcam</label>
                <div className="flex">
                  {!webcamActive ? (
                    <Button
                      primary
                      onClick={startWebcam}
                    >
                      <Camera size={18} className="inline mr-2" />
                      Start
                    </Button>
                  ) : (
                    <Button
                      onClick={stopWebcam}
                      className="bg-red-600 hover:bg-red-700"
                    >
                      Stop
                    </Button>
                  )}
                </div>
              </div>

            </div>
          </Panel>
        )}

        {/* Photo Detection Results */}
        {activeTab === 'photo' && imagePreview && (
          <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
            {/* Left: Original Image */}
            <Panel title="Original Image" className="h-full flex flex-col">
              <div 
                className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border min-h-0"
                style={{ borderColor: 'var(--border-primary)', background: 'var(--bg-secondary)' }}
              >
                <img 
                  src={imagePreview}
                  alt="Original"
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            </Panel>

            {/* Right: Detection Result */}
            <Panel title="Detection Results" className="h-full flex flex-col">
              {result ? (
                <div className="h-full flex flex-col min-h-0">
                  <div className="flex items-center justify-between mb-4 shrink-0">
                    <div className="flex items-center gap-2">
                      <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                      <span 
                        className="font-mono"
                        style={{ 
                          color: result.success && result.result?.success ? 'var(--success-text)' : 'var(--error-text)' 
                        }}
                      >
                        {result.success && result.result?.success ? 'SEGMENTED' : 'ERROR'}
                      </span>
                    </div>
                    {result.result?.inference_time && (
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {(result.result.inference_time * 1000).toFixed(0)}ms
                      </span>
                    )}
                  </div>

                  {result.success && result.result?.success ? (
                    <>
                      {result.result.annotated_image ? (
                        <div className="flex-1 flex flex-col min-h-0 relative group">
                          <div 
                            className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border relative"
                            style={{ borderColor: 'var(--border-primary)', background: 'var(--bg-secondary)' }}
                          >
                            <img 
                              src={`${BASE_URL}/predictions/${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                              alt="Detection Result"
                              className="max-w-full max-h-full object-contain"
                              onError={(e) => {
                                console.error('Failed to load image:', e);
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                            <a 
                              href={`${BASE_URL}/predictions/${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                              download={`predicted_${result.result.annotated_image.replace(/\\/g, '/').split('/').pop()}`}
                              className="absolute bottom-4 right-4 p-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 shadow-lg border"
                              style={{ 
                                background: 'var(--bg-card)', 
                                borderColor: 'var(--border-primary)',
                                color: 'var(--text-secondary)'
                              }}
                              title="Download Image"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              <Download size={16} />
                              <span className="text-xs font-medium">Download</span>
                            </a>
                          </div>
                        </div>
                      ) : (
                        <div 
                          className="flex-1 flex items-center justify-center min-h-0"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          No annotated image available
                        </div>
                      )}

                      <div className="mt-4 shrink-0">
                        <div 
                          className="flex items-center justify-between text-sm mb-2"
                          style={{ color: 'var(--text-secondary)' }}
                        >
                          <span>{result.result.num_detections} {result.result.num_detections === 1 ? 'crack' : 'cracks'} detected</span>
                          <span 
                            className="text-xs px-2 py-1 rounded"
                            style={{ color: 'var(--accent-primary)', background: 'var(--accent-glow)' }}
                          >
                            Analysis
                          </span>
                        </div>
                        
                        {result.result.num_detections > 0 && result.result.detections && (
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div 
                              className="p-2 rounded flex flex-col"
                              style={{ background: 'var(--bg-tertiary)' }}
                            >
                              <span style={{ color: 'var(--text-muted)' }}>Average Confidence:</span>
                              <span style={{ color: 'var(--text-secondary)' }}>
                                {(result.result.detections.reduce((acc: number, d: any) => acc + d.confidence, 0) / result.result.num_detections * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div 
                              className="p-2 rounded flex flex-col"
                              style={{ background: 'var(--bg-tertiary)' }}
                            >
                              <span style={{ color: 'var(--text-muted)' }}>Total Crack Area:</span>
                              <span style={{ color: 'var(--text-secondary)' }}>
                                {result.result.detections.reduce((acc: number, d: any) => acc + (d.area || 0), 0).toLocaleString(undefined, { maximumFractionDigits: 0 })} px²
                              </span>
                            </div>

                          </div>
                        )}
                      </div>
                    </>
                  ) : (
                    <div 
                      className="flex-1 flex items-center justify-center min-h-0"
                      style={{ color: 'var(--error-text)' }}
                    >
                      {result.result?.error || result.error || 'An unknown error occurred'}
                    </div>
                  )}
                </div>
              ) : (
                <div 
                  className="h-full flex flex-col items-center justify-center min-h-0"
                  style={{ color: 'var(--text-muted)' }}
                >
                  <Upload size={48} className="mb-4" style={{ opacity: 0.3 }} />
                  <p>Click "Run Detection" to see results</p>
                  <p className="text-sm mt-2">The annotated image will appear here</p>
                </div>
              )}
            </Panel>
          </div>
        )}

        {/* Video Detection Results */}
        {activeTab === 'video' && videoPreview && (
          <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
            {/* Left: Original Video */}
            <Panel title="Original Video" className="h-full flex flex-col">
              <div 
                className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border min-h-0"
                style={{ borderColor: 'var(--border-primary)', background: 'var(--bg-secondary)' }}
              >
                <video 
                  src={videoPreview}
                  controls
                  className="max-w-full max-h-full"
                />
              </div>
            </Panel>

            {/* Right: Detection Result */}
            <Panel title="Detection Results" className="h-full flex flex-col">
              {result ? (
                <div className="h-full flex flex-col min-h-0">
                  <div className="flex items-center justify-between mb-4 shrink-0">
                    <div className="flex items-center gap-2">
                      <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                      <span 
                        className="font-mono"
                        style={{ 
                          color: result.success && result.result?.success ? 'var(--success-text)' : 'var(--error-text)' 
                        }}
                      >
                        {result.success && result.result?.success ? 'PROCESSED' : 'ERROR'}
                      </span>
                    </div>
                  </div>

                  {result.success && result.result?.success ? (
                    <>
                      {result.result.annotated_video ? (
                        <div 
                          className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border min-h-0 relative group"
                          style={{ borderColor: 'var(--border-primary)', background: 'var(--bg-secondary)' }}
                        >
                          <video 
                            src={`${BASE_URL}/predictions/${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                            controls
                            className="max-w-full max-h-full"
                          />
                          <a 
                            href={`${BASE_URL}/predictions/${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                            download={`predicted_${result.result.annotated_video.replace(/\\/g, '/').split('/').pop()}`}
                            className="absolute bottom-4 right-4 p-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 shadow-lg border z-10"
                            style={{
                              background: 'var(--bg-card)',
                              borderColor: 'var(--border-primary)',
                              color: 'var(--text-secondary)'
                            }}
                            title="Download Video"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <Download size={16} />
                            <span className="text-xs font-medium">Download</span>
                          </a>
                        </div>
                      ) : (
                        <div 
                          className="flex-1 flex items-center justify-center min-h-0"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          No annotated video available
                        </div>
                      )}

                      <div 
                        className="mt-4 space-y-2 text-sm shrink-0"
                        style={{ color: 'var(--text-secondary)' }}
                      >
                        <div className="flex items-center justify-between">
                          <span>Total detections: {result.result.total_detections}</span>
                          <span 
                            className="text-xs px-2 py-1 rounded"
                            style={{ color: 'var(--accent-primary)', background: 'var(--accent-glow)' }}
                          >
                            Video Segmentation
                          </span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div 
                      className="flex-1 flex items-center justify-center min-h-0"
                      style={{ color: 'var(--error-text)' }}
                    >
                      {result.result?.error || result.error || 'Video processing failed'}
                    </div>
                  )}
                </div>
              ) : (
                <div 
                  className="h-full flex flex-col items-center justify-center min-h-0"
                  style={{ color: 'var(--text-muted)' }}
                >
                  <Video size={48} className="mb-4" style={{ opacity: 0.3 }} />
                  <p>Click "Run Detection" to process video</p>
                  <p className="text-sm mt-2">The annotated video will appear here after processing</p>
                </div>
              )}
            </Panel>
          </div>
        )}

        {/* Webcam Results - Single Live Feed with Real-time Detection */}
        {activeTab === 'webcam' && webcamActive && (
          <div className="h-full" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
            <Panel title="Live Webcam Detection" className="h-full flex flex-col relative overflow-hidden">
              <div 
                className="flex-1 relative min-h-0 rounded-lg overflow-hidden border"
                style={{ borderColor: 'var(--border-primary)', background: 'var(--bg-secondary)' }}
              >
                <video 
                  ref={webcamVideoRef}
                  autoPlay 
                  playsInline
                  className="absolute inset-0 w-full h-full object-contain"
                />
                <DetectionOverlay 
                  videoRef={webcamVideoRef}
                  detections={detections}
                  isActive={webcamActive}
                />
                
                {/* Stats Overlay */}
                <div className="absolute top-4 left-4 flex flex-col gap-2 z-20">
                  <div 
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg border backdrop-blur-sm"
                    style={{ background: 'var(--bg-card)', borderColor: 'var(--border-primary)' }}
                  >
                    <LED color="green" pulse />
                    <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>LIVE</span>
                  </div>
                  {fps > 0 && (
                    <div 
                      className="px-3 py-1 rounded-lg border text-[10px] font-mono backdrop-blur-sm"
                      style={{ background: 'var(--bg-card)', borderColor: 'var(--border-primary)', color: 'var(--success-text)' }}
                    >
                      {fps} FPS
                    </div>
                  )}
                </div>

                {/* Detection Count Overlay */}
                {detections.length > 0 && (
                  <div className="absolute top-4 right-4 z-20">
                    <div 
                      className="px-3 py-1.5 rounded-lg border text-[10px] font-mono backdrop-blur-sm"
                      style={{ background: 'var(--bg-card)', borderColor: 'var(--border-primary)', color: 'var(--success-text)' }}
                    >
                      {detections.length} {detections.length === 1 ? 'crack' : 'cracks'} detected
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 flex items-center gap-4 shrink-0">
                <div className="flex items-center gap-2">
                  <span 
                    className="text-xs uppercase font-bold tracking-wider"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Real-time Detection
                  </span>
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ 
                      background: autoCapture ? 'var(--success-text)' : 'var(--text-disabled)',
                      animation: autoCapture ? 'pulse 2s infinite' : undefined
                    }}
                  />
                </div>
                <button
                  onClick={() => setAutoCapture(!autoCapture)}
                  className="px-3 py-1 rounded-lg text-xs font-medium transition-all border"
                  style={
                    autoCapture 
                      ? { 
                          background: 'var(--success-bg)', 
                          color: 'var(--success-text)', 
                          borderColor: 'var(--success-text)'
                        }
                      : { 
                          background: 'var(--bg-tertiary)', 
                          color: 'var(--text-muted)', 
                          borderColor: 'var(--border-secondary)'
                        }
                  }
                >
                  {autoCapture ? 'ON' : 'OFF'}
                </button>
              </div>
            </Panel>
          </div>
        )}
      </div>
    </div>
  );
};

export default Detection;
