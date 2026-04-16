import { useEffect } from 'react';
import './ConceptDialog.css';

export default function ReferencesDialog({ onClose }) {
    // Close on Escape key
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleEscape);
        return () => window.removeEventListener('keydown', handleEscape);
    }, [onClose]);

    return (
        <div className="dialog-overlay" onClick={onClose}>
            <div className="dialog-content concept-dialog" onClick={(e) => e.stopPropagation()}>
                <button className="dialog-close" onClick={onClose} aria-label="Close dialog">
                    ×
                </button>

                <div className="concept-dialog-body">
                    <h2 className="concept-title">References</h2>

                    <div className="concept-explanation">
                        <ul style={{ lineHeight: '2' }}>
                            <li>
                                <a 
                                    href="https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Neural Networks from Scratch - Tsoding
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://www.ibm.com/think/machine-learning" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    IBM: Machine Learning
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://xkcd.com/1838/" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    xkcd: Machine Learning
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Training Progress Simple Explanation Image - Jeremy Jordan
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2017-11-07-at-12.32.19-PM.png" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Matrix Multiplications Image - Jeremy Jordan
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://nsweb.tn.tudelft.nl/~gsteele/TN2513_2019_2020/3%20Differentiation.html#Overview-of-finite-difference-methods" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Finite Difference - Gary Steele's Old TU-Delft Website
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://media.licdn.com/dms/image/v2/D5612AQECXl0DLimxgQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1726329287502?e=2147483647&v=beta&t=T-f2b7Omn7oQylTvfrBu5zNw-xlIrizWLCnPoO1Jpcc" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Backpropagation Simple Explanation - Indeera Weerasinghe
                                </a>
                            </li>
                            <li>
                                <a 
                                    href="https://i0.wp.com/sefiks.com/wp-content/uploads/2020/02/sample-activation-functions-square.png" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    Activation Functions Image - Sefik Ilkin Serengil
                                </a>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
}
