import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [query, setQuery] = useState('');
  const [queryStatus, setQueryStatus] = useState('idle');
  const [queryResult, setQueryResult] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback(handleFileDrop, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false
  });

  function handleFileDrop(acceptedFiles) {
    if (acceptedFiles.length > 0) {
      uploadFile(acceptedFiles[0]);
    }
  }

  async function uploadFile(file) {
    setUploadStatus('uploading');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Uploading file:', file.name, 'Size:', file.size);
      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });
      
      console.log('Upload response:', response.data);

      // Backend returns job_id, file_id, status, and file_info
      setUploadedFile({
        filename: response.data.file_info.filename,
        size: response.data.file_info.size,
        id: response.data.file_id
      });
      setSessionId(response.data.file_id);
      setUploadStatus('uploading'); // Keep as uploading until job completes
      
      // Poll for job completion
      pollJobStatus(response.data.job_id);
    } catch (error) {
      console.error('Upload error:', error);
      console.error('Error details:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        config: error.config
      });
      setUploadStatus('error');
      // Log more details about the error
      if (error.response) {
        console.error('Error response:', error.response.data);
      }
    }
  }

  async function pollJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/status/${jobId}`);
        
        if (response.data.status === 'completed') {
          setUploadStatus('completed');
          setUploadProgress(100);
          clearInterval(pollInterval);
        } else if (response.data.status === 'failed') {
          setUploadStatus('error');
          clearInterval(pollInterval);
        } else if (response.data.status === 'processing') {
          // Update progress if available
          if (response.data.progress) {
            setUploadProgress(response.data.progress);
          }
        }
      } catch (error) {
        console.error('Job status check error:', error);
        setUploadStatus('error');
        clearInterval(pollInterval);
      }
    }, 1000);
  }

  async function submitQuery() {
    if (!query.trim() || !sessionId) return;

    setIsProcessing(true);
    setQueryStatus('processing');
    setQueryResult(null);

    try {
      const response = await axios.post(`${API_BASE}/api/query`, {
        file_id: sessionId,
        query: query
      });

      const queryId = response.data.query_id;
      
      // Poll for query result
      pollQueryResult(queryId);
    } catch (error) {
      console.error('Query error:', error);
      setQueryStatus('error');
      setIsProcessing(false);
    }
  }

  async function pollQueryResult(queryId) {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/query/${queryId}`);
        
        if (response.data.status === 'completed') {
          setQueryResult(response.data.result);
          setQueryStatus('completed');
          
          // Add to conversation history
          setConversationHistory(prev => [...prev, {
            query: query,
            result: response.data.result,
            timestamp: new Date().toLocaleTimeString()
          }]);
          
          setQuery('');
          setIsProcessing(false);
          clearInterval(pollInterval);
        } else if (response.data.status === 'error') {
          console.error('Query error:', response.data.error || 'Unknown error');
          setQueryStatus('error');
          setIsProcessing(false);
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Query result check error:', error);
        setQueryStatus('error');
        setIsProcessing(false);
        clearInterval(pollInterval);
      }
    }, 1000);
  }

  function resetUpload() {
    setUploadedFile(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setSessionId(null);
    setConversationHistory([]);
    setQueryResult(null);
    setQueryStatus('idle');
    setIsProcessing(false);
  }

  return (
    <div className="App">
      <div className="header">
        <h1>Agentic CSV QA</h1>
        <p>Upload CSV files and ask questions in natural language</p>
      </div>

      <div className="main-container">
        {/* File Upload Section */}
        {!uploadedFile && (
          <div className="upload-section">
            <div className="upload-card">
              <h2>Upload CSV File</h2>
              
              {uploadStatus === 'idle' && (
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                  <input {...getInputProps()} />
                  {isDragActive ? (
                    <p>Drop the CSV file here...</p>
                  ) : (
                    <div>
                      <p>Drag & drop a CSV file here, or click to select</p>
                      <p style={{ fontSize: '0.9rem', color: '#666' }}>
                        Only CSV files are supported
                      </p>
                    </div>
                  )}
                </div>
              )}

              {uploadStatus === 'uploading' && (
                <div>
                  <p>Uploading file...</p>
                  <div style={{ width: '100%', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
                    <div 
                      style={{ 
                        width: `${uploadProgress}%`, 
                        height: '20px', 
                        backgroundColor: '#667eea',
                        borderRadius: '4px',
                        transition: 'width 0.3s ease'
                      }}
                    />
                  </div>
                  <p>{uploadProgress}% complete</p>
                </div>
              )}

              {uploadStatus === 'completed' && uploadedFile && (
                <div>
                  <div className="status completed">
                    ✅ File uploaded successfully!
                  </div>
                  <p><strong>File:</strong> {uploadedFile.filename}</p>
                  <p><strong>Size:</strong> {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '1rem' }}>
                    🚀 Ready to analyze! Click "Start Chat" to begin asking questions.
                  </p>
                  <button className="btn" onClick={() => setUploadedFile(uploadedFile)}>
                    Start Chat
                  </button>
                  <button className="btn btn-secondary" onClick={resetUpload} style={{ marginLeft: '0.5rem' }}>
                    Upload New File
                  </button>
                </div>
              )}

              {uploadStatus === 'error' && (
                <div>
                  <div className="status error">
                    ❌ Upload failed. Please try again.
                  </div>
                  <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
                    Make sure you're uploading a valid CSV file (less than 50MB).
                  </p>
                  <button className="btn" onClick={resetUpload}>
                    Try Again
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Chat Interface */}
        {uploadedFile && (
          <div className="chat-container">
            <div className="chat-header">
              <div className="file-info">
                <span className="file-icon">📄</span>
                <span className="file-name">{uploadedFile.filename}</span>
                <button className="btn btn-small" onClick={resetUpload}>
                  Change File
                </button>
              </div>
            </div>

            <div className="chat-messages">
              {/* Welcome Message */}
              <div className="message assistant">
                <div className="message-content">
                  <p>Hello! I'm here and ready to help you analyze your CSV data. What would you like to know about your dataset?</p>
                  <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.5rem' }}>
                    You can ask questions like:
                  </p>
                  <ul style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                    <li>"What are the column names?"</li>
                    <li>"Show me a summary of the data"</li>
                    <li>"What are the data types?"</li>
                    <li>"Show me the first 5 rows"</li>
                    <li>"What are the most common values in column X?"</li>
                    <li>"Calculate the average of column Y"</li>
                  </ul>
                  <p style={{ fontSize: '0.9rem', color: '#667eea', marginTop: '1rem', fontStyle: 'italic' }}>
                    💡 Tip: I can analyze any CSV file without knowing what the data represents. I focus on structure and patterns!
                  </p>
                </div>
              </div>

              {/* Conversation History */}
              {conversationHistory.map((item, index) => (
                <div key={index}>
                  {/* User Message */}
                  <div className="message user">
                    <div className="message-content">
                      {item.query}
                    </div>
                  </div>

                  {/* Assistant Response */}
                  <div className="message assistant">
                    <div className="message-content">
                      {item.result && item.result.success ? (
                        <div>
                          <div style={{ 
                            backgroundColor: '#f8f9fa', 
                            padding: '1rem', 
                            borderRadius: '8px',
                            marginBottom: '0.5rem',
                            whiteSpace: 'pre-wrap',
                            fontFamily: 'monospace',
                            fontSize: '0.9rem'
                          }}>
                            {item.result.result || 'No result available'}
                          </div>
                          {item.result.execution_time && (
                            <p style={{ fontSize: '0.8rem', color: '#666' }}>
                              Execution time: {item.result.execution_time.toFixed(2)}s
                            </p>
                          )}
                        </div>
                      ) : (
                        <div style={{ 
                          backgroundColor: '#f8d7da', 
                          color: '#721c24', 
                          padding: '1rem', 
                          borderRadius: '8px',
                          border: '1px solid #f5c6cb'
                        }}>
                          <strong>Error:</strong> {item.result?.error || 'Unknown error occurred'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Current Query Result */}
              {queryResult && (
                <div className="message assistant">
                  <div className="message-content">
                    {queryResult.success ? (
                      <div>
                        <div style={{ 
                          backgroundColor: '#f8f9fa', 
                          padding: '1rem', 
                          borderRadius: '8px',
                          marginBottom: '0.5rem',
                          whiteSpace: 'pre-wrap',
                          fontFamily: 'monospace',
                          fontSize: '0.9rem'
                        }}>
                          {queryResult.result || 'No result available'}
                        </div>
                        {queryResult.execution_time && (
                          <p style={{ fontSize: '0.8rem', color: '#666' }}>
                            Execution time: {queryResult.execution_time.toFixed(2)}s
                          </p>
                        )}
                      </div>
                    ) : (
                      <div style={{ 
                        backgroundColor: '#f8d7da', 
                        color: '#721c24', 
                        padding: '1rem', 
                        borderRadius: '8px',
                        border: '1px solid #f5c6cb'
                      }}>
                        <strong>Error:</strong> {queryResult.error || 'Unknown error occurred'}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Processing Indicator */}
              {isProcessing && (
                <div className="message assistant">
                  <div className="message-content">
                    <div className="processing-indicator">
                      <div className="loading-spinner"></div>
                      <span>Analyzing your data...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Section */}
            <div className="chat-input-container">
              <div className="input-wrapper">
                <textarea
                  className="chat-input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a question about your data..."
                  rows="1"
                  disabled={isProcessing}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (!isProcessing && query.trim()) {
                        submitQuery();
                      }
                    }
                  }}
                />
                <button 
                  className="send-button"
                  onClick={submitQuery}
                  disabled={!query.trim() || isProcessing}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                  </svg>
                </button>
              </div>
              <div className="input-footer">
                <span style={{ fontSize: '0.8rem', color: '#666' }}>
                  Press Enter to send, Shift+Enter for new line
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
