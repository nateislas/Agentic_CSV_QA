import React, { useState } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [queryStatus, setQueryStatus] = useState('idle');
  const [sessionId, setSessionId] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);

  // API base URL
  const API_BASE = 'http://localhost:8000';

  // File upload dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv']
    },
    onDrop: handleFileDrop
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

      setUploadedFile({
        fileId: response.data.file_id,
        jobId: response.data.job_id,
        filename: file.name,
        size: file.size
      });

      // Start polling for job status
      pollJobStatus(response.data.job_id);
    } catch (error) {
      console.error('Upload error:', error);
      console.error('Error details:', error.response?.data);
      setUploadStatus('error');
      setUploadProgress(0);
    }
  }

  async function pollJobStatus(jobId) {
    console.log('Starting to poll job status for:', jobId);
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/status/${jobId}`);
        console.log('Job status response:', response.data);
        
        if (response.data.status === 'completed') {
          setUploadStatus('completed');
          setUploadProgress(100);
          clearInterval(pollInterval);
          
          // Get file ID for queries
          if (response.data.result && response.data.result.file_id) {
            setSessionId(response.data.result.file_id);
          }
        } else if (response.data.status === 'failed') {
          setUploadStatus('error');
          setUploadProgress(0);
          clearInterval(pollInterval);
        } else {
          setUploadProgress(response.data.progress || 0);
        }
      } catch (error) {
        console.error('Status check error:', error);
        console.error('Error details:', error.response?.data);
        setUploadStatus('error');
        clearInterval(pollInterval);
      }
    }, 1000);
  }

  async function submitQuery() {
    if (!query.trim() || !sessionId) return;

    setQueryStatus('processing');
    setQueryResult(null);

    try {
      const response = await axios.post(`${API_BASE}/api/query`, {
        file_id: sessionId,
        query: query
        // Don't send session_id - let the backend create a new session
      });

      const queryId = response.data.query_id;
      
      // Poll for query result
      pollQueryResult(queryId);
    } catch (error) {
      console.error('Query error:', error);
      setQueryStatus('error');
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
          clearInterval(pollInterval);
        } else if (response.data.status === 'error') {
          setQueryStatus('error');
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Query result check error:', error);
        setQueryStatus('error');
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
  }

  return (
    <div className="App">
      <div className="header">
        <h1>Agentic CSV QA</h1>
        <p>Upload CSV files and ask questions in natural language</p>
      </div>

      <div className="container">
        <div className="grid">
          {/* File Upload Section */}
          <div className="card">
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
                <button className="btn btn-secondary" onClick={resetUpload}>
                  Upload New File
                </button>
              </div>
            )}

            {uploadStatus === 'error' && (
              <div>
                <div className="status error">
                  ❌ Upload failed. Please try again.
                </div>
                <button className="btn" onClick={resetUpload}>
                  Try Again
                </button>
              </div>
            )}
          </div>

          {/* Query Section */}
          <div className="card">
            <h2>Ask Questions</h2>
            
            {!sessionId ? (
              <p>Please upload a CSV file first to start asking questions.</p>
            ) : (
              <div>
                <div className="form-group">
                  <label htmlFor="query">Ask a question about your data:</label>
                  <textarea
                    id="query"
                    className="form-control"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g., 'Show me the average sales by region' or 'What are the top 10 products?'"
                    rows="3"
                    disabled={queryStatus === 'processing'}
                  />
                </div>
                
                <button 
                  className="btn" 
                  onClick={submitQuery}
                  disabled={!query.trim() || queryStatus === 'processing'}
                >
                  {queryStatus === 'processing' ? (
                    <>
                      <span className="loading"></span> Processing...
                    </>
                  ) : (
                    'Ask Question'
                  )}
                </button>

                {queryResult && (
                  <div className="result">
                    <h3>Result:</h3>
                    <pre>{JSON.stringify(queryResult, null, 2)}</pre>
                  </div>
                )}

                {queryStatus === 'error' && (
                  <div className="status error">
                    ❌ Query processing failed. Please try again.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Conversation History */}
        {conversationHistory.length > 0 && (
          <div className="card">
            <h2>Conversation History</h2>
            {conversationHistory.map((item, index) => (
              <div key={index} style={{ marginBottom: '1rem', padding: '1rem', border: '1px solid #e9ecef', borderRadius: '4px' }}>
                <div style={{ marginBottom: '0.5rem' }}>
                  <strong>Q:</strong> {item.query}
                  <span style={{ fontSize: '0.8rem', color: '#666', marginLeft: '1rem' }}>
                    {item.timestamp}
                  </span>
                </div>
                <div>
                  <strong>A:</strong> 
                  <pre style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
                    {JSON.stringify(item.result, null, 2)}
                  </pre>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
