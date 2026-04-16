/**
 * Collision detection and resolution for React Flow nodes
 * Treats group nodes as single units and moves children together
 */

/**
 * Get bounding boxes from nodes, filtering to only top-level nodes (no parentId)
 * @param {Array} nodes - Array of React Flow nodes
 * @param {number} margin - Additional margin around nodes
 * @returns {Array} Array of box objects with position, size, and node reference
 */
function getBoxesFromNodes(nodes, margin = 0) {
  // Only process top-level nodes (those without a parentId)
  const topLevelNodes = nodes.filter(node => !node.parentId);
  const boxes = [];

  for (const node of topLevelNodes) {
    // Get node dimensions from various possible sources
    const width = node.width ?? node.measured?.width ?? node.style?.width ?? 0;
    const height = node.height ?? node.measured?.height ?? node.style?.height ?? 0;

    // Neural network group has highest priority (shouldn't move)
    const priority = node.id === 'neural-network' ? 1000 : 1;

    boxes.push({
      x: node.position.x - margin,
      y: node.position.y - margin,
      width: width + margin * 2,
      height: height + margin * 2,
      node,
      priority,
      moved: false,
    });
  }

  return boxes;
}

/**
 * Find all descendant nodes (children, grandchildren, etc.)
 * @param {Array} nodes - Array of all nodes
 * @param {string} parentId - ID of the parent node
 * @returns {Array} Array of descendant nodes
 */
function findDescendants(nodes, parentId) {
  const descendants = [];
  const directChildren = nodes.filter(n => n.parentId === parentId);
  
  for (const child of directChildren) {
    descendants.push(child);
    // Recursively find nested children
    descendants.push(...findDescendants(nodes, child.id));
  }
  
  return descendants;
}

/**
 * Resolve collisions between nodes using iterative physics-based separation
 * Groups are treated as single units and children move with parents
 * 
 * @param {Array} nodes - Array of React Flow nodes
 * @param {Object} options - Configuration options
 * @param {number} options.maxIterations - Maximum collision resolution iterations
 * @param {number} options.overlapThreshold - Minimum overlap to trigger resolution
 * @param {number} options.margin - Additional spacing between nodes
 * @returns {Array} Array of nodes with updated positions
 */
export function resolveCollisions(
  nodes,
  { maxIterations = 50, overlapThreshold = 0.5, margin = 0 } = {}
) {
  const boxes = getBoxesFromNodes(nodes, margin);
  
  // Early exit if no top-level nodes
  if (boxes.length <= 1) {
    return nodes;
  }

  // Iteratively resolve collisions
  for (let iter = 0; iter < maxIterations; iter++) {
    let moved = false;

    // Check all pairs of nodes for collisions
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const A = boxes[i];
        const B = boxes[j];

        // Calculate center positions
        const centerAX = A.x + A.width * 0.5;
        const centerAY = A.y + A.height * 0.5;
        const centerBX = B.x + B.width * 0.5;
        const centerBY = B.y + B.height * 0.5;

        // Calculate distance between centers
        const dx = centerAX - centerBX;
        const dy = centerAY - centerBY;

        // Calculate overlap along each axis
        const px = (A.width + B.width) * 0.5 - Math.abs(dx);
        const py = (A.height + B.height) * 0.5 - Math.abs(dy);

        // Check if there's significant overlap
        if (px > overlapThreshold && py > overlapThreshold) {
          moved = true;
          
          // Use priority to determine who moves
          // Higher priority = less movement (neural network stays put)
          const totalPriority = A.priority + B.priority;
          const weightA = B.priority / totalPriority; // A moves proportional to B's priority
          const weightB = A.priority / totalPriority; // B moves proportional to A's priority
          
          // Resolve along the smallest overlap axis (minimum separation)
          if (px < py) {
            // Move along x-axis
            const sx = dx > 0 ? 1 : -1;
            const moveA = (px * weightA) * sx;
            const moveB = (px * weightB) * sx;
            A.x += moveA;
            B.x -= moveB;
            if (moveA !== 0) A.moved = true;
            if (moveB !== 0) B.moved = true;
          } else {
            // Move along y-axis
            const sy = dy > 0 ? 1 : -1;
            const moveA = (py * weightA) * sy;
            const moveB = (py * weightB) * sy;
            A.y += moveA;
            B.y -= moveB;
            if (moveA !== 0) A.moved = true;
            if (moveB !== 0) B.moved = true;
          }
        }
      }
    }

    // Early exit if no overlaps were found
    if (!moved) {
      break;
    }
  }

  // Build a map of position changes for parent nodes
  const positionDeltas = new Map();
  
  for (const box of boxes) {
    if (box.moved) {
      const oldX = box.node.position.x;
      const oldY = box.node.position.y;
      const newX = box.x + margin;
      const newY = box.y + margin;
      
      positionDeltas.set(box.node.id, {
        dx: newX - oldX,
        dy: newY - oldY,
        newX,
        newY,
      });
    }
  }

  // Apply position changes to all nodes (parents and children)
  const newNodes = nodes.map((node) => {
    // If this is a parent that moved, update its position
    if (positionDeltas.has(node.id)) {
      const { newX, newY } = positionDeltas.get(node.id);
      return {
        ...node,
        position: { x: newX, y: newY },
      };
    }
    
    // If this is a child whose parent moved, apply the delta
    if (node.parentId && positionDeltas.has(node.parentId)) {
      const { dx, dy } = positionDeltas.get(node.parentId);
      return {
        ...node,
        position: {
          x: node.position.x + dx,
          y: node.position.y + dy,
        },
      };
    }
    
    // Check for nested parents (grandparent moved)
    // Walk up the parent chain and accumulate deltas
    let currentParentId = node.parentId;
    let totalDx = 0;
    let totalDy = 0;
    let hasAncestorMoved = false;
    
    while (currentParentId) {
      if (positionDeltas.has(currentParentId)) {
        const { dx, dy } = positionDeltas.get(currentParentId);
        totalDx += dx;
        totalDy += dy;
        hasAncestorMoved = true;
      }
      
      // Find the parent node to continue walking up the chain
      const parentNode = nodes.find(n => n.id === currentParentId);
      currentParentId = parentNode?.parentId;
    }
    
    if (hasAncestorMoved) {
      return {
        ...node,
        position: {
          x: node.position.x + totalDx,
          y: node.position.y + totalDy,
        },
      };
    }

    // Node didn't move
    return node;
  });

  return newNodes;
}
