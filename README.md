Run it                                                                                                        
                                                                                                                
  1. Install deps:                                                                                              
                                                                                                                
  uv sync                                                                                                       
                                                                                                                
  2. Build RAG documents + vector DB from your dataset:                                                         
                                                                                                                
  uv run python -m app.scripts.build_vectorstore                                                                
                                                                                                                
  3. Start API:                                                                                                 
                                                                                                                
  uv run run-app                                                                                                
                                                                                                                
  Test it                                                                                                       
                                                                                                                
  - Health:                                                                                                     
                                                                                                                
  Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -Method GET                                             
                                                                                                                
  - Retrieval only:                                                                                             
                                                                                                                
  $body = @{                                                                                                    
    query = "What other structures are found in Banteay Chhmar?"                                                
    top_k = 5                                                                                                   
  } | ConvertTo-Json                                                                                            
  Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/retrieve-only" -Method POST -ContentType "application/js 
  on" -Body $body                                                                                               

  - Retrieval only:

  $body = @{
    query = "What other structures are found in Banteay Chhmar?"
    top_k = 5
  } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/retrieve-only" -Method POST -ContentType "application/js
  on" -Body $body

  - Full RAG answer:

  $body = @{
    question = "Besides the main temple complex, what other structures are found within Banteay Chhmar's
  grounds?"
    top_k = 5
  } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/v1/query" -Method POST -ContentType "application/json" -Body
  $body