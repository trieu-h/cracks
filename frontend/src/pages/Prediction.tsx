import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, RotateCcw, Image, Video } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runPrediction, runVideoPrediction, getModels, BASE_URL } from '../api';

const Prediction: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'photo' | 'video'>('photo');
  
  // Photo prediction states
  const [imagePath, setImagePath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // Video prediction states
  const [videoPath, setVideoPath] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [sampleInterval, setSampleInterval] = useState<number>(5);
  
  // Common states
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models, setModels] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string>('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

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
    } else {
      videoInputRef.current?.click();
    }
  };

  const handlePredict = async () => {
    const selectedModelData = models.find(m => m.id === selectedModel);
    
    if (activeTab === 'photo') {
      if (!selectedFile || !selectedModel) return;

      setLoading(true);
      setProgress('');

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('model_path', selectedModelData?.path || '');
      formData.append('conf', '0.25');

      try {
        const res = await runPrediction(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run prediction'
        });
      }
    } else {
      if (!selectedVideo || !selectedModel) return;

      setLoading(true);
      setProgress('Extracting frames... This may take a while.');

      const formData = new FormData();
      formData.append('video', selectedVideo);
      formData.append('model_path', selectedModelData?.path || '');
      formData.append('conf', '0.25');
      formData.append('sample_interval', sampleInterval.toString());

      try {
        const res = await runVideoPrediction(formData);
        setResult(res.data);
      } catch (error) {
        setResult({
          success: false,
          error: 'Failed to run video prediction'
        });
      }
    }
    setLoading(false);
    setProgress('');
  };

  return (
    <div className="space-y-6 h-full overflow-auto">
      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-stone-700 pb-4">
        <button
          onClick={() => setActiveTab('photo')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'photo' 
              ? 'bg-stone-700 text-white' 
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
          }`}
        >
          <Image size={18} />
          Photo
        </button>
        <button
          onClick={() => setActiveTab('video')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'video' 
              ? 'bg-stone-700 text-white' 
              : 'text-stone-400 hover:text-stone-200 hover:bg-stone-800'
          }`}
        >
          <Video size={18} />
          Video
        </button>
      </div>

      {/* Configuration Panel */}
      <Panel title={`${activeTab === 'photo' ? 'Photo' : 'Video'} Prediction Configuration`}>
        <div className="grid grid-cols-2 gap-4">
          {/* Model Selection */}
          <div>
            <label className="text-sm text-stone-400 mb-2 block">Select Model</label>
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
              <p className="text-xs text-stone-500 mt-2">
                No trained models available. Train a model first.
              </p>
            )}
          </div>

          {/* File Upload */}
          <div>
            <label className="text-sm text-stone-400 mb-2 block">
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
                className="px-4 py-2 bg-stone-700 hover:bg-stone-600 rounded-xl text-sm text-stone-200 transition-all"
                title="Browse files"
              >
                Browse
              </button>
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
                    Run Prediction
                  </>
                )}
              </Button>
            </div>
            {activeTab === 'video' && (
              <div className="mt-2">
                <label className="text-xs text-stone-400">Frame Sample Interval (lower = more accurate but slower)</label>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={sampleInterval}
                  onChange={(e) => setSampleInterval(parseInt(e.target.value))}
                  className="w-full mt-1"
                />
                <div className="flex justify-between text-xs text-stone-500 mt-1">
                  <span>1 (all frames)</span>
                  <span>{sampleInterval} frame(s)</span>
                  <span>30 (fast)</span>
                </div>
              </div>
            )}
            {progress && (
              <p className="text-xs text-blue-400 mt-2 animate-pulse">{progress}</p>
            )}
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
          </div>
        </div>
      </Panel>

      {/* Photo Prediction Results */}
      {activeTab === 'photo' && imagePreview && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
          {/* Left: Original Image */}
          <Panel title="Original Image" className="h-full flex flex-col">
            <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
              <img 
                src={imagePreview}
                alt="Original"
                className="max-w-full max-h-full object-contain"
              />
            </div>
          </Panel>

          {/* Right: Prediction Result */}
          <Panel title="Detection Results" className="h-full flex flex-col">
            {result ? (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-2">
                    <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                    <span className={`font-mono ${result.success && result.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                      {result.success && result.result?.success ? 'SEGMENTED' : 'ERROR'}
                    </span>
                  </div>
                  {result.result?.inference_time && (
                    <span className="text-xs text-stone-500">
                      {(result.result.inference_time * 1000).toFixed(0)}ms
                    </span>
                  )}
                </div>

                {result.success && result.result?.success ? (
                  <>
                    {result.result.annotated_image ? (
                      <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
                        <img 
                          src={`${BASE_URL}/predictions/${result.result.annotated_image.split('/').pop()}`}
                          alt="Detection Result"
                          className="max-w-full max-h-full object-contain"
                          onError={(e) => {
                            console.error('Failed to load image:', e);
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-stone-500 min-h-0">
                        No annotated image available
                      </div>
                    )}

                    <div className="mt-4 text-sm text-stone-400 shrink-0 flex items-center justify-between">
                      <span>{result.result.num_detections} objects segmented</span>
                      <span className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">Segmentation</span>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                    {result.result?.error || result.error || 'An unknown error occurred'}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                <Upload size={48} className="mb-4 opacity-30" />
                <p>Click "Run Prediction" to see results</p>
                <p className="text-sm mt-2">The annotated image will appear here</p>
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Video Prediction Results */}
      {activeTab === 'video' && videoPreview && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 320px)', minHeight: '400px' }}>
          {/* Left: Original Video */}
          <Panel title="Original Video" className="h-full flex flex-col">
            <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
              <video 
                src={videoPreview}
                controls
                className="max-w-full max-h-full"
              />
            </div>
          </Panel>

          {/* Right: Prediction Result */}
          <Panel title="Detection Results" className="h-full flex flex-col">
            {result ? (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-2">
                    <LED color={result.success && result.result?.success ? 'green' : 'red'} />
                    <span className={`font-mono ${result.success && result.result?.success ? 'text-green-400' : 'text-red-400'}`}>
                      {result.success && result.result?.success ? 'PROCESSED' : 'ERROR'}
                    </span>
                  </div>
                </div>

                {result.success && result.result?.success ? (
                  <>
                    {result.result.annotated_video ? (
                      <div className="flex-1 flex items-center justify-center rounded-lg overflow-hidden border border-stone-700 bg-stone-900/50 min-h-0">
                        <video 
                          src={`${BASE_URL}/predictions/${result.result.annotated_video.split('/').pop()}`}
                          controls
                          className="max-w-full max-h-full"
                        />
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-stone-500 min-h-0">
                        No annotated video available
                      </div>
                    )}

                    <div className="mt-4 space-y-2 text-sm text-stone-400 shrink-0">
                      <div className="flex items-center justify-between">
                        <span>Total detections: {result.result.total_detections}</span>
                        <span className="text-xs text-blue-400 bg-blue-900/20 px-2 py-1 rounded">Video Segmentation</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Frames processed:</span>
                          <span className="ml-2 text-stone-300">{result.result.processed_frames}/{result.result.total_frames}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Frames with cracks:</span>
                          <span className="ml-2 text-stone-300">{result.result.frames_with_detections}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Sample interval:</span>
                          <span className="ml-2 text-stone-300">{result.result.sample_interval}</span>
                        </div>
                        <div className="bg-stone-800/50 p-2 rounded">
                          <span className="text-stone-500">Processing time:</span>
                          <span className="ml-2 text-stone-300">{result.result.total_time?.toFixed(1)}s</span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-red-400 min-h-0">
                    {result.result?.error || result.error || 'An unknown error occurred'}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-stone-500 min-h-0">
                <Video size={48} className="mb-4 opacity-30" />
                <p>Click "Run Prediction" to see results</p>
                <p className="text-sm mt-2">The annotated video will appear here</p>
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Empty State for Photo */}
      {activeTab === 'photo' && !imagePreview && (
        <div className="flex flex-col items-center justify-center h-96 text-stone-500">
          <Upload size={64} className="mb-4 opacity-30" />
          <p className="text-lg">Select an image to start</p>
          <p className="text-sm mt-2">Choose a photo and click "Browse" to get started</p>
        </div>
      )}

      {/* Empty State for Video */}
      {activeTab === 'video' && !videoPreview && (
        <div className="flex flex-col items-center justify-center h-96 text-stone-500">
          <Video size={64} className="mb-4 opacity-30" />
          <p className="text-lg">Select a video to start</p>
          <p className="text-sm mt-2">Choose a video and click "Browse" to get started</p>
        </div>
      )}
    </div>
  );
};

export default Prediction;
