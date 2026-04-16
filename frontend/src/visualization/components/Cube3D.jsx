import { useRef } from 'react';
import './Cube3D.css';

export default function Cube3D({ onClick, isExpanded = false }) {
  const svgRef = useRef(null);

  // === Isometric projection parameters ===
  const size = 80;
  const angle = Math.PI / 6; // 30°
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  // === Cube vertices (centered at origin) ===
  const vertices = [
    [-1, -1, -1], // 0 back-bottom-left
    [1, -1, -1], // 1 back-bottom-right
    [1, 1, -1], // 2 back-top-right
    [-1, 1, -1], // 3 back-top-left
    [-1, -1, 1], // 4 front-bottom-left
    [1, -1, 1], // 5 front-bottom-right
    [1, 1, 1], // 6 front-top-right
    [-1, 1, 1], // 7 front-top-left
  ];

  // === Projection ===
  const project = (x, y, z) => {
    const scale = size / 2;
    const px = (x - z) * cos * scale;
    const py = (x + z) * sin * scale - y * scale;
    return { x: px + size, y: py + size };
  };

  // === Faces ===
  const faces = [
    { indices: [0, 1, 2, 3], color: '#1a1a1a' }, // back
    { indices: [4, 5, 6, 7], color: '#2a2a2a' }, // front
    { indices: [0, 1, 5, 4], color: '#222' },    // bottom
    { indices: [2, 3, 7, 6], color: '#2f2f2f' }, // top
    { indices: [0, 3, 7, 4], color: '#1f1f1f' }, // left
    { indices: [1, 2, 6, 5], color: '#252525' }, // right
  ];

  // === Painter’s algorithm (simple depth sort) ===
  const sortedFaces = faces
    .map(face => ({
      ...face,
      zSum: face.indices.reduce((s, i) => s + vertices[i][2], 0),
    }))
    .sort((a, b) => b.zSum - a.zSum);

  // === Visible (solid) edges ===

  const visibleEdges = [
    [1, 2],
    [2, 3],

    [4, 5],
    [5, 6],
    [6, 7],

    [7, 4],
    [3, 7],
    [6, 2],
    [5, 1],
  ];


  // === Hidden (dotted) edges ===
  const hiddenEdges = [
    [0, 4], // back-bottom-left → front-bottom-left
    [0, 1], // back-bottom-right → front-bottom-right
    [0, 3], // back-top-left → front-top-left
  ];

  return (
    <div
      className={`cube-3d-container ${isExpanded ? 'expanded' : ''}`}
      onClick={onClick}
    >
      <svg
        ref={svgRef}
        width={size * 2}
        height={size * 2}
        viewBox={`0 0 ${size * 2} ${size * 2}`}
        className="cube-3d-svg"
        onClick={onClick}
        style={{ pointerEvents: 'all', cursor: 'pointer' }}
      >
        {/* Faces */}
        {sortedFaces.map((face, i) => {
          const points = face.indices
            .map(idx => {
              const [x, y, z] = vertices[idx];
              const p = project(x, y, z);
              return `${p.x},${p.y}`;
            })
            .join(' ');

          return (
            <polygon
              key={`face-${i}`}
              points={points}
              fill={face.color}
              stroke="none"
            />
          );
        })}

        {/* Hidden edges (dotted) */}
        {hiddenEdges.map((edge, i) => {
          const [a, b] = edge;
          const p1 = project(...vertices[a]);
          const p2 = project(...vertices[b]);

          return (
            <line
              key={`hidden-${i}`}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke="#555"
              strokeWidth="1"
              strokeDasharray="3 3"
              opacity="0.7"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}

        {/* Visible edges (solid) */}
        {visibleEdges.map((edge, i) => {
          const [a, b] = edge;
          const p1 = project(...vertices[a]);
          const p2 = project(...vertices[b]);

          return (
            <line
              key={`visible-${i}`}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke="#777"
              strokeWidth="1.5"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      {!isExpanded && (
        <div className="cube-label">Explore more</div>
      )}
    </div>
  );
}
