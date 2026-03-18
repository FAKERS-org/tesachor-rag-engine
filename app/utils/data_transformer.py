"""
Transform conversation JSONL into RAG-ready documents
Extracts facts from assistant responses, preserves context
"""

import json
import re
from typing import List, Dict, Iterator
from dataclasses import dataclass
from pathlib import Path

@dataclass
class KnowledgeChunk:
    """Structured chunk for vector storage"""
    content: str           # What gets embedded
    metadata: Dict         # Filterable attributes
    source_conversation: str  # Original thread ID
    turn_number: int       # Position in conversation
    
class ConversationTransformer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_jsonl_files(self) -> Iterator[Dict]:
        """Load all _part_*.jsonl files"""
        for file_path in sorted(self.data_dir.glob("_part_*.jsonl")):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        record['_source_file'] = file_path.name
                        record['_line_number'] = line_num
                        yield record
                    except json.JSONDecodeError:
                        continue
    
    def extract_facts(self, record: Dict) -> List[KnowledgeChunk]:
        """
        Convert conversation thread into fact chunks
        Strategy: Each assistant response → standalone fact document
        """
        messages = record.get('messages', [])
        chunks = []
        
        # Find conversation topic from first user message
        topic = self._extract_topic(messages)
        conversation_id = f"{record['_source_file']}_{record['_line_number']}"
        
        # Track conversation flow for context
        conversation_history = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant' and i > 0:
                # Get preceding user question
                user_msg = messages[i-1] if messages[i-1]['role'] == 'user' else None
                
                if user_msg:
                    # Create enriched content: Question + Answer + Context
                    content = self._create_fact_document(
                        question=user_msg['content'],
                        answer=msg['content'],
                        topic=topic,
                        history=conversation_history[-3:]  # Last 3 turns for context
                    )
                    
                    chunk = KnowledgeChunk(
                        content=content,
                        metadata={
                            "topic": topic,
                            "question_type": self._classify_question(user_msg['content']),
                            "has_numbers": self._contains_numbers(msg['content']),
                            "location_mentioned": self._extract_locations(msg['content']),
                            "conversation_id": conversation_id,
                            "turn": i // 2,  # Turn number in conversation
                            "source_file": record['_source_file']
                        },
                        source_conversation=conversation_id,
                        turn_number=i
                    )
                    chunks.append(chunk)
                    
                    # Update history for next iteration
                    conversation_history.append({
                        'q': user_msg['content'],
                        'a': msg['content']
                    })
        
        return chunks
    
    def _create_fact_document(
        self, 
        question: str, 
        answer: str, 
        topic: str,
        history: List[Dict]
    ) -> str:
        """
        Create standalone fact document from Q-A pair
        Includes context so it makes sense without conversation history
        """
        # Clean up conversational filler
        clean_answer = self._clean_response(answer)
        clean_question = self._clean_question(question)
        
        # Build context-aware document
        parts = [f"Topic: {topic}"]
        
        # Add conversation context if relevant (for follow-up questions)
        if history and self._is_follow_up(clean_question):
            context = " | ".join([
                f"Previously: {h['q']} → {h['a'][:100]}..."
                for h in history[-2:]
            ])
            parts.append(f"Context: {context}")
        
        parts.extend([
            f"Question: {clean_question}",
            f"Answer: {clean_answer}"
        ])
        
        return " | ".join(parts)
    
    def _clean_response(self, text: str) -> str:
        """Remove conversational filler words"""
        fillers = [
            "Hi there!", "Hello!", "I'd be delighted", "Feel free to ask",
            "This is one of my favorite topics!", "Absolutely!", "Great question!"
        ]
        result = text
        for filler in fillers:
            result = result.replace(filler, "").strip()
        return result
    
    def _clean_question(self, text: str) -> str:
        """Normalize question format"""
        # Remove polite prefixes
        text = re.sub(r'^(Can you|Could you|Please|I\'d like to know|I want to know)\s+', '', text, flags=re.IGNORECASE)
        # Remove question marks for embedding consistency
        text = text.rstrip('?').strip()
        return text
    
    def _extract_topic(self, messages: List[Dict]) -> str:
        """Identify main topic from conversation"""
        # Look for location names in first few messages
        locations = ['Banteay Chhmar', 'Angkor', 'Phnom Penh', 'Siem Reap', 
                    'Battambang', 'Kampot', 'Kep', 'Sihanoukville']
        
        text = " ".join([m['content'] for m in messages[:3]])
        
        for loc in locations:
            if loc.lower() in text.lower():
                return loc
        
        # Fallback: classify by content
        if any(word in text.lower() for word in ['temple', 'wat', 'pagoda']):
            return "Cambodian Temples"
        elif any(word in text.lower() for word in ['economy', 'gdp', 'macro']):
            return "Cambodian Economy"
        
        return "General Cambodia Tourism"
    
    def _classify_question(self, question: str) -> str:
        """Classify question type for metadata filtering"""
        q = question.lower()
        
        if any(w in q for w in ['what is', 'what are', 'define']):
            return "definition"
        elif any(w in q for w in ['how', 'way to', 'process']):
            return "process"
        elif any(w in q for w in ['where', 'location', 'near']):
            return "location"
        elif any(w in q for w in ['when', 'time', 'hours', 'schedule']):
            return "temporal"
        elif any(w in q for w in ['why', 'reason', 'cause']):
            return "causal"
        elif any(w in q for w in ['size', 'how big', 'how much', 'many', 'number']):
            return "quantitative"
        else:
            return "general"
    
    def _contains_numbers(self, text: str) -> bool:
        """Check if answer contains numerical data"""
        return bool(re.search(r'\d+', text))
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract mentioned locations"""
        cambodia_locations = [
            'Banteay Chhmar', 'Angkor Wat', 'Phnom Penh', 'Siem Reap',
            'Battambang', 'Kampot', 'Kep', 'Sihanoukville', 'Koh Rong',
            'Tonle Sap', 'Mekong', 'Bassac', 'Sisowath Quay'
        ]
        found = []
        text_lower = text.lower()
        for loc in cambodia_locations:
            if loc.lower() in text_lower:
                found.append(loc)
        return found
    
    def _is_follow_up(self, question: str) -> bool:
        """Detect if question references previous context"""
        indicators = ['it', 'that', 'the temple', 'there', 'this place', 
                     'they', 'them', 'those', 'the area']
        return any(ind in question.lower() for ind in indicators)
    
    def transform_all(self, output_file: str = "data/rag_documents.jsonl"):
        """Process all files and save transformed data"""
        all_chunks = []
        
        for record in self.load_jsonl_files():
            chunks = self.extract_facts(record)
            all_chunks.extend(chunks)
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps({
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'conversation_id': chunk.source_conversation,
                    'turn': chunk.turn_number
                }, ensure_ascii=False) + '\n')
        
        print(f"Transformed {len(all_chunks)} chunks from conversations")
        print(f"Saved to {output_file}")
        
        # Print sample
        if all_chunks:
            print("\nSample chunk:")
            print(all_chunks[0].content[:300] + "...")
        
        return all_chunks

# Run transformation
if __name__ == "__main__":
    transformer = ConversationTransformer()
    chunks = transformer.transform_all()