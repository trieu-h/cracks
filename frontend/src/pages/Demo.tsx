import React, { useState } from 'react';
import { Image, X } from 'lucide-react';

// Demo items with actual images
const demoItems = [
  {
    id: 1,
    type: 'image',
    title: 'Concrete Surface Crack',
    description: 'Structural crack detection with high confidence',
    src: '/demo/pred_b077a492.jpg'
  },
  {
    id: 2,
    type: 'image',
    title: 'Asphalt Pavement Damage',
    description: 'Road surface crack identification',
    src: '/demo/pred_91d92072.jpg'
  },
  {
    id: 3,
    type: 'image',
    title: 'Infrastructure Assessment',
    description: 'Detailed crack analysis and segmentation',
    src: '/demo/pred_8d8f9081.jpg'
  },
  {
    id: 4,
    type: 'video',
    title: 'Highway Inspection',
    description: 'Real-time crack detection on asphalt pavement',
    src: '/demo/video_pred_f7f8a325.mp4'
  },
  {
    id: 5,
    type: 'video',
    title: 'Infrastructure Survey',
    description: 'Automated road condition assessment',
    src: '/demo/video_pred_579e350b.mp4'
  }
];

const Demo: React.FC = () => {
  const [selectedItem, setSelectedItem] = useState<typeof demoItems[0] | null>(null);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-serif" style={{ color: 'var(--text-primary)' }}>
          Detection Showcase
        </h2>
        <p style={{ color: 'var(--text-muted)' }}>
          Sample crack detection results on various infrastructure
        </p>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-3 gap-4">
        {demoItems.map((item) => (
          <div
            key={item.id}
            className="group relative rounded-xl overflow-hidden cursor-pointer"
            style={{ 
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-primary)'
            }}
            onClick={() => setSelectedItem(item)}
          >
            {/* Media Area */}
            <div className="aspect-video relative">
              {item.src ? (
                item.type === 'video' ? (
                  // Video - autoplay in grid
                  <video
                    src={item.src}
                    className="w-full h-full object-cover"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                ) : (
                  // Image
                  <img
                    src={item.src}
                    alt={item.title}
                    className="w-full h-full object-cover"
                  />
                )
              ) : (
                // Placeholder
                <div 
                  className="w-full h-full flex items-center justify-center"
                  style={{ background: 'var(--bg-tertiary)' }}
                >
                  <div 
                    className="w-16 h-16 rounded-xl flex items-center justify-center"
                    style={{ 
                      background: 'var(--success-bg)',
                      border: '2px solid var(--success-text)'
                    }}
                  >
                    <Image size={24} style={{ color: 'var(--success-text)' }} />
                  </div>
                </div>
              )}

              {/* Hover Overlay */}
              <div 
                className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ background: 'rgba(0, 0, 0, 0.5)' }}
              >
                <span className="text-white font-medium">
                  {item.type === 'video' ? 'Click to enlarge' : 'Click to view'}
                </span>
              </div>

              {/* Type Badge */}
              <div 
                className="absolute top-3 left-3 px-2 py-1 rounded text-xs font-medium uppercase"
                style={{ 
                  background: 'var(--bg-card)',
                  color: 'var(--text-muted)'
                }}
              >
                {item.type}
              </div>
            </div>

            {/* Info */}
            <div className="p-4">
              <h3 
                className="font-medium mb-1 truncate"
                style={{ color: 'var(--text-secondary)' }}
              >
                {item.title}
              </h3>
              <p 
                className="text-sm truncate"
                style={{ color: 'var(--text-muted)' }}
              >
                {item.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal */}
      {selectedItem && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-8"
          style={{ background: 'rgba(0, 0, 0, 0.9)' }}
          onClick={() => setSelectedItem(null)}
        >
          <div 
            className="max-w-5xl w-full rounded-xl overflow-hidden"
            style={{ background: 'var(--bg-secondary)' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Media */}
            <div className="relative flex items-center justify-center" style={{ maxHeight: '70vh' }}>
              {selectedItem.src ? (
                selectedItem.type === 'video' ? (
                  <video
                    src={selectedItem.src}
                    controls
                    autoPlay
                    className="w-full max-h-[70vh] object-contain"
                  />
                ) : (
                  <img
                    src={selectedItem.src}
                    alt={selectedItem.title}
                    className="w-full h-auto max-h-[70vh] object-contain"
                  />
                )
              ) : (
                <div 
                  className="aspect-video flex items-center justify-center"
                  style={{ background: 'var(--bg-tertiary)' }}
                >
                  <div 
                    className="w-24 h-24 rounded-xl flex items-center justify-center"
                    style={{ 
                      background: 'var(--success-bg)',
                      border: '3px solid var(--success-text)'
                    }}
                  >
                    <Image size={40} style={{ color: 'var(--success-text)' }} />
                  </div>
                </div>
              )}

              {/* Close Button */}
              <button
                onClick={() => setSelectedItem(null)}
                className="absolute top-4 right-4 w-10 h-10 rounded-full flex items-center justify-center transition-colors"
                style={{ 
                  background: 'var(--bg-card)',
                  color: 'var(--text-secondary)'
                }}
              >
                <X size={24} />
              </button>
            </div>

            {/* Modal Info */}
            <div className="p-6">
              <h3 
                className="text-xl font-medium mb-2"
                style={{ color: 'var(--text-secondary)' }}
              >
                {selectedItem.title}
              </h3>
              <p style={{ color: 'var(--text-muted)' }}>
                {selectedItem.description}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Demo;
