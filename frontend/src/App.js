import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

// Development mode debugging
const isDevelopment = process.env.NODE_ENV === 'development';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileId, setFileId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [query, setQuery] = useState('');
  const [queryStatus, setQueryStatus] = useState('idle');
  const [queryResult, setQueryResult] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Polling registry to track active intervals
  const pollingRegistry = useRef(new Map());
  const hasProcessedQuery = useRef(false); // Track if we've already processed this query
  const submitQueryCallCount = useRef(0); // Track submitQuery calls for debugging

  // Cleanup all polling intervals on unmount
  useEffect(() => {
    return () => {
      pollingRegistry.current.forEach((interval) => {
        clearInterval(interval);
      });
      pollingRegistry.current.clear();
    };
  }, []);

  // Helper function to clear polling for a specific query
  const clearPollingForQuery = useCallback((queryId) => {
    const interval = pollingRegistry.current.get(queryId);
    if (interval) {
      clearInterval(interval);
      pollingRegistry.current.delete(queryId);
      if (isDevelopment) {
        console.log(`Cleared polling for query: ${queryId}`);
        console.log(`Active polling intervals: ${pollingRegistry.current.size}`);
      }
    }
  }, []);

  // Helper function to check if we're already polling a query
  const isPollingQuery = useCallback((queryId) => {
    return pollingRegistry.current.has(queryId);
  }, []);

  // Debug function to log polling state
  const logPollingState = useCallback(() => {
    if (isDevelopment) {
      console.log('=== Polling State Debug ===');
      console.log('Active intervals:', pollingRegistry.current.size);
      console.log('Interval keys:', Array.from(pollingRegistry.current.keys()));
      console.log('Has processed query:', hasProcessedQuery.current);
      console.log('Submit query call count:', submitQueryCallCount.current);
      console.log('==========================');
    }
  }, []);

  // Helper function to format numbers
  const formatNumber = (num) => {
    if (typeof num === 'number') {
      if (Number.isInteger(num)) {
        return num.toLocaleString();
      } else {
        return num.toFixed(2);
      }
    }
    return String(num);
  };

  // Helper function to truncate text
  const truncateText = (text, maxLength = 100) => {
    if (typeof text === 'string' && text.length > maxLength) {
      return text.substring(0, maxLength) + '...';
    }
    return text;
  };

  // Reusable component for displaying results
  const renderResult = (result, resultType, metadata = {}) => {
    // 1. TABLE RESULTS
    if (resultType === 'table' && Array.isArray(result)) {
      const columns = result.length > 0 ? Object.keys(result[0]) : [];
      const totalRows = result.length;
      const displayRows = result.slice(0, 20); // Show first 20 rows
      
      return (
        <div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <h4 style={{ margin: '0', color: '#2c3e50' }}>
              📊 Table Results
            </h4>
            <span style={{ 
              fontSize: '0.8rem', 
              color: '#666',
              backgroundColor: '#e9ecef',
              padding: '0.2rem 0.5rem',
              borderRadius: '4px'
            }}>
              {totalRows} row{totalRows !== 1 ? 's' : ''}
            </span>
          </div>
          
          <div style={{ 
            maxHeight: '400px', 
            overflowY: 'auto',
            border: '1px solid #dee2e6',
            borderRadius: '6px'
          }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse', 
              fontSize: '0.8rem',
              backgroundColor: 'white'
            }}>
              <thead>
                <tr style={{ 
                  backgroundColor: '#f8f9fa',
                  position: 'sticky',
                  top: 0,
                  zIndex: 1
                }}>
                  {columns.map(key => (
                    <th key={key} style={{ 
                      padding: '12px 8px', 
                      borderBottom: '2px solid #dee2e6', 
                      textAlign: 'left',
                      fontWeight: '600',
                      color: '#495057'
                    }}>
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayRows.map((row, index) => (
                  <tr key={index} style={{
                    backgroundColor: index % 2 === 0 ? 'white' : '#f8f9fa'
                  }}>
                    {columns.map((key, cellIndex) => {
                      const value = row[key];
                      return (
                        <td key={cellIndex} style={{ 
                          padding: '8px', 
                          borderBottom: '1px solid #dee2e6',
                          maxWidth: '200px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {value !== null && value !== undefined ? 
                            (typeof value === 'number' ? formatNumber(value) : String(value)) : 
                            <span style={{ color: '#6c757d', fontStyle: 'italic' }}>—</span>
                          }
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {totalRows > 20 && (
            <p style={{ 
              fontSize: '0.8rem', 
              color: '#666', 
              marginTop: '0.5rem',
              textAlign: 'center'
            }}>
              Showing first 20 rows of {totalRows} total rows
            </p>
          )}
        </div>
      );
    }
    
    // 2. TEXT RESULTS
    else if (resultType === 'text' || typeof result === 'string') {
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            📝 Analysis Result
          </h4>
          <div style={{
            lineHeight: '1.6',
            whiteSpace: 'pre-wrap'
          }}>
            {result}
          </div>
        </div>
      );
    }
    
    // 3. ARRAY/LIST RESULTS
    else if (Array.isArray(result)) {
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            📋 List Results ({result.length} items)
          </h4>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <ul style={{ 
              margin: '0', 
              paddingLeft: '1.5rem',
              lineHeight: '1.6'
            }}>
              {result.slice(0, 50).map((item, index) => (
                <li key={index} style={{ marginBottom: '0.25rem' }}>
                  {typeof item === 'object' ? 
                    JSON.stringify(item) : 
                    String(item)
                  }
                </li>
              ))}
            </ul>
            {result.length > 50 && (
              <p style={{ 
                fontSize: '0.8rem', 
                color: '#666', 
                marginTop: '0.5rem'
              }}>
                Showing first 50 items of {result.length} total items
              </p>
            )}
          </div>
        </div>
      );
    }
    
    // 4. DICTIONARY/OBJECT RESULTS
    else if (typeof result === 'object' && result !== null) {
      const entries = Object.entries(result);
      
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            🔍 Analysis Results
          </h4>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse',
              fontSize: '0.9rem'
            }}>
              <tbody>
                {entries.map(([key, value], index) => (
                  <tr key={index} style={{
                    backgroundColor: index % 2 === 0 ? 'white' : '#f8f9fa'
                  }}>
                    <td style={{ 
                      padding: '8px', 
                      fontWeight: '600',
                      color: '#495057',
                      borderBottom: '1px solid #dee2e6',
                      width: '30%'
                    }}>
                      {key}
                    </td>
                    <td style={{ 
                      padding: '8px', 
                      borderBottom: '1px solid #dee2e6'
                    }}>
                      {typeof value === 'number' ? 
                        formatNumber(value) : 
                        (typeof value === 'object' ? 
                          JSON.stringify(value) : 
                          String(value)
                        )
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    
    // 5. NUMBER RESULTS
    else if (typeof result === 'number') {
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            🔢 Analysis Result
          </h4>
          <div style={{
            fontSize: '1.2rem',
            fontWeight: '600',
            color: '#2c3e50',
            padding: '1rem',
            backgroundColor: '#e8f5e8',
            borderRadius: '6px',
            textAlign: 'center'
          }}>
            {formatNumber(result)}
          </div>
        </div>
      );
    }
    
    // 6. PLOT/CHART RESULTS
    else if (resultType === 'plot') {
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            📈 Visualization Generated
          </h4>
          <div style={{
            padding: '1rem',
            backgroundColor: '#e8f4fd',
            borderRadius: '6px',
            textAlign: 'center',
            color: '#0c5460'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
            <p style={{ margin: '0' }}>
              A chart or plot has been generated for your data.
            </p>
            {metadata.figure_count && (
              <p style={{ 
                fontSize: '0.8rem', 
                margin: '0.5rem 0 0 0',
                color: '#6c757d'
              }}>
                Generated {metadata.figure_count} figure{metadata.figure_count !== 1 ? 's' : ''}
              </p>
            )}
          </div>
        </div>
      );
    }
    
    // 7. FALLBACK FOR UNKNOWN TYPES
    else {
      return (
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#2c3e50' }}>
            📄 Analysis Result
          </h4>
          <div style={{
            backgroundColor: '#fff3cd',
            border: '1px solid #ffeaa7',
            borderRadius: '6px',
            padding: '1rem',
            fontSize: '0.9rem'
          }}>
            <pre style={{ 
              margin: '0', 
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word'
            }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>
      );
    }
  };

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
      setFileId(response.data.file_id);
      // Generate a unique session ID for this conversation
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
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

  const submitQuery = useCallback(async () => {
    submitQueryCallCount.current += 1;
    const callId = submitQueryCallCount.current;
    
    if (isDevelopment) {
      console.log(`submitQuery called (call #${callId}) with:`, { 
        query: query, 
        sessionId, 
        isProcessing,
        timestamp: new Date().toISOString()
      });
    }
    
    if (!query.trim() || !fileId || !sessionId || isProcessing) {
      if (isDevelopment) {
        console.log(`submitQuery early return (call #${callId}):`, { 
          hasQuery: !!query.trim(), 
          hasFileId: !!fileId,
          hasSessionId: !!sessionId, 
          isProcessing 
        });
      }
      return;
    }

    // Reset the processed flag for new query
    hasProcessedQuery.current = false;

    if (isDevelopment) {
      console.log(`Starting query submission (call #${callId})`);
      logPollingState();
    }
    
    setIsProcessing(true);
    setQueryStatus('processing');
    setQueryResult(null);

    try {
      const response = await axios.post(`${API_BASE}/api/query`, {
        file_id: fileId,
        session_id: sessionId,
        query: query
      });

      const queryId = response.data.query_id;
      if (isDevelopment) {
        console.log(`Query submitted (call #${callId}), got queryId:`, queryId);
      }
      
      // Poll for query result
      pollQueryResult(queryId, query);
    } catch (error) {
      console.error('Query error:', error);
      setQueryStatus('error');
      setIsProcessing(false);
    }
  }, [query, fileId, sessionId, isProcessing, logPollingState]);

  const pollQueryResult = useCallback(async (queryId, queryText) => {
    if (isDevelopment) {
      console.log(`Starting poll for query: ${queryId}`);
      logPollingState();
    }
    
    // Don't start polling if we're already polling this query
    if (isPollingQuery(queryId)) {
      if (isDevelopment) {
        console.log(`Already polling query: ${queryId}, skipping`);
      }
      return;
    }
    
    // Add timeout mechanism to prevent infinite polling
    const maxPollingTime = 5 * 60 * 1000; // 5 minutes
    const startTime = Date.now();
    let pollCount = 0;
    
    const pollInterval = setInterval(async () => {
      pollCount++;
      
      // Check if we've exceeded the maximum polling time
      if (Date.now() - startTime > maxPollingTime) {
        if (isDevelopment) {
          console.log(`Query ${queryId} polling timeout after ${pollCount} attempts`);
        }
        
        if (!hasProcessedQuery.current) {
          hasProcessedQuery.current = true;
          setQueryStatus('error');
          setIsProcessing(false);
        }
        
        clearPollingForQuery(queryId);
        return;
      }
      
      if (isDevelopment) {
        console.log(`Polling query: ${queryId} (attempt #${pollCount})`);
      }
      
      try {
        const response = await axios.get(`${API_BASE}/api/query/${queryId}`);
        if (isDevelopment) {
          console.log(`Query ${queryId} status:`, response.data.status);
        }
        
        if (response.data.status === 'completed') {
          if (isDevelopment) {
            console.log(`Query ${queryId} completed, clearing interval`);
          }
          
          // Only process the result once
          if (!hasProcessedQuery.current) {
            hasProcessedQuery.current = true;
            
            // Use functional state updates to ensure atomic operations
            setQueryResult(response.data.result);
            setQueryStatus('completed');
            
            // Add to conversation history using functional update
            setConversationHistory(prev => {
              // Check if this conversation entry already exists to prevent duplicates
              const entryExists = prev.some(entry => 
                entry.query === queryText && 
                entry.timestamp === new Date().toLocaleTimeString()
              );
              
              if (entryExists) {
                if (isDevelopment) {
                  console.log('Conversation entry already exists, skipping duplicate');
                }
                return prev;
              }
              
              return [...prev, {
                query: queryText,
                result: response.data.result,
                timestamp: new Date().toLocaleTimeString()
              }];
            });
            
            // Clear query text using functional update
            setQuery(prev => {
              if (prev === queryText) {
                return '';
              }
              return prev;
            });
            
            setIsProcessing(false);
            
            // Clear the queryResult after a short delay to prevent duplicate display
            setTimeout(() => {
              setQueryResult(null);
            }, 100);
          }
          
          clearPollingForQuery(queryId);
        } else if (response.data.status === 'error') {
          console.error('Query error:', response.data.error || 'Unknown error');
          
          if (!hasProcessedQuery.current) {
            hasProcessedQuery.current = true;
            setQueryStatus('error');
            setIsProcessing(false);
          }
          
          clearPollingForQuery(queryId);
        }
      } catch (error) {
        console.error('Query result check error:', error);
        
        if (!hasProcessedQuery.current) {
          hasProcessedQuery.current = true;
          setQueryStatus('error');
          setIsProcessing(false);
        }
        
        clearPollingForQuery(queryId);
      }
    }, 1000);
    
    // Register the polling interval
    pollingRegistry.current.set(queryId, pollInterval);
    if (isDevelopment) {
      console.log(`Registered polling for query: ${queryId}`);
      logPollingState();
    }
  }, [isPollingQuery, clearPollingForQuery, logPollingState]); // Dependencies for helper functions

  function resetUpload() {
    // Clear all polling intervals when resetting
    pollingRegistry.current.forEach((interval) => {
      clearInterval(interval);
    });
    pollingRegistry.current.clear();
    hasProcessedQuery.current = false;
    submitQueryCallCount.current = 0;
    
    setUploadedFile(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setFileId(null);
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
                            fontSize: '0.9rem'
                          }}>
                            {renderResult(item.result.result, item.result.result_type, item.result.metadata)}
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
                          fontSize: '0.9rem'
                                                  }}>
                            {renderResult(queryResult.result, queryResult.result_type, queryResult.metadata)}
                        </div>
                        {queryResult.execution_time && (
                          <p style={{ fontSize: '0.8rem', color: '#666' }}>
                            ⏱️ Execution time: {queryResult.execution_time.toFixed(2)}s
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
                        <strong>❌ Error:</strong> {queryResult.error || 'Unknown error occurred'}
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
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    // Prevent multiple rapid clicks
                    if (!isProcessing && query.trim()) {
                      submitQuery();
                    }
                  }}
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
