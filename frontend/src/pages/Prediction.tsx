import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, RotateCcw } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { LED } from '../components/ui/LED';
import { runPrediction, getModels, BASE_URL } from '../api';

const Prediction: React.FC = () => {
  const [imagePath, setImagePath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [models, setModels] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getModels().then(res => setModels(res.data));
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setImagePath(file.name);
      setImagePreview(URL.createObjectURL(file));
      setResult(null); // Clear previous result when selecting new image
    }
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const handlePredict = async () => {
    if (!selectedFile || !selectedModel) return;

    setLoading(true);
    const selectedModelData = models.find(m => m.id === selectedModel);

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
    setLoading(false);
  };

  return (
    <div className="space-y-6 h-full overflow-auto">
      {/* Configuration Panel */}
      <Panel title="Prediction Configuration">
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

          {/* Image Upload */}
          <div>
            <label className="text-sm text-stone-400 mb-2 block">Select Image</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={imagePath}
                readOnly
                placeholder="Click Browse to select an image..."
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
                disabled={loading || !selectedFile || !selectedModel}
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
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>
      </Panel>

      {/* Image Comparison */}
      {imagePreview && (
        <div className="grid grid-cols-2 gap-6" style={{ height: 'calc(100vh - 280px)', minHeight: '400px' }}>
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
                      {result.success && result.result?.success ? 'DETECTED' : 'ERROR'}
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

                    <div className="mt-4 text-sm text-stone-400 shrink-0">
                      {result.result.num_detections} objects detected
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

      {/* Empty State */}
      {!imagePreview && (
        <div className="flex flex-col items-center justify-center h-96 text-stone-500">
          <Upload size={64} className="mb-4 opacity-30" />
          <p className="text-lg">Select an image to start</p>
          <p className="text-sm mt-2">Choose a photo and click "Browse" to get started</p>
        </div>
      )}
    </div>
  );
};

export default Prediction;
