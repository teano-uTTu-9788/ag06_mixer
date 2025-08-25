#!/usr/bin/env python3
"""
Production NLP System
Following Hugging Face Transformers and Meta AI best practices
"""

import asyncio
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# Configure logging following Google Cloud practices
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Voice command intents following Meta AI classification standards"""
    VOLUME_ADJUST = "volume_adjust"
    PAN_ADJUST = "pan_adjust"
    EQ_ADJUST = "eq_adjust"
    EFFECT_CONTROL = "effect_control"
    MUTE_TOGGLE = "mute_toggle"
    SOLO_TOGGLE = "solo_toggle"
    RECORD_CONTROL = "record_control"
    SCENE_RECALL = "scene_recall"
    QUERY_STATUS = "query_status"
    AUTOMATION_CONTROL = "automation_control"
    UNKNOWN = "unknown"

class EntityType(Enum):
    """Named entities in audio commands"""
    CHANNEL = "channel"
    PARAMETER = "parameter"
    VALUE = "value"
    DIRECTION = "direction"
    EFFECT = "effect"
    SCENE = "scene"

@dataclass
class Entity:
    """Named entity extraction result"""
    entity_type: EntityType
    value: Any
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class ParsedCommand:
    """Parsed voice command following structured format"""
    raw_text: str
    intent: IntentType
    intent_confidence: float
    entities: List[Entity]
    timestamp: datetime
    session_id: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogContext:
    """Conversation context management"""
    session_id: str
    last_command: Optional[ParsedCommand] = None
    last_channel: Optional[int] = None
    command_history: deque = field(default_factory=lambda: deque(maxlen=10))
    user_preferences: Dict[str, Any] = field(default_factory=dict)

class ProductionNLP:
    """
    Production NLP System for Audio Mixing Commands
    Following Hugging Face and Meta AI best practices
    """
    
    def __init__(self):
        # Intent classification patterns (regex-based for reliability)
        self.intent_patterns = {
            IntentType.VOLUME_ADJUST: [
                r'(?:set|change|adjust|make|turn)\s+(?:the\s+)?(?:volume|level|gain)\s+(?:of\s+)?(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?\s+(?:to|at)?\s*([\d.]+|louder|quieter|up|down)',
                r'(?:turn|bring)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?\s*(up|down|louder|quieter)',
                r'(?:increase|decrease|boost|lower)\s+(?:the\s+)?(?:volume|level)\s+(?:of\s+)?(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
            ],
            IntentType.PAN_ADJUST: [
                r'(?:pan|move|place)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?\s+(?:to\s+(?:the\s+)?)?(left|right|center|middle)',
                r'(?:set|adjust)\s+(?:the\s+)?pan(?:ning)?\s+(?:of\s+)?(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?\s+(?:to\s+)?(left|right|center|[\d.]+)',
            ],
            IntentType.EQ_ADJUST: [
                r'(?:boost|cut|adjust|add|remove)\s+(?:the\s+)?(bass|mid|treble|highs?|lows?|mids?)\s+(?:on\s+)?(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
                r'(?:eq|equalize)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?\s+(?:at\s+)?(\d+(?:k|hz)?)\s+(?:by\s+)?([\+\-]?\d+(?:db)?)?',
            ],
            IntentType.EFFECT_CONTROL: [
                r'(?:add|apply|enable|turn\s+on)\s+(reverb|delay|chorus|compression|compressor|gate)\s+(?:to|on)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
                r'(?:remove|disable|turn\s+off)\s+(reverb|delay|chorus|compression|compressor|gate)\s+(?:from|on)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
            ],
            IntentType.MUTE_TOGGLE: [
                r'(?:mute|unmute|silence)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar|all)?',
            ],
            IntentType.SOLO_TOGGLE: [
                r'(?:solo|unsolo|isolate)\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
                r'(?:listen\s+to\s+)?(?:only\s+)?(?:the\s+)?(\d+|vocals?|drums?|bass|guitar)',
            ],
            IntentType.SCENE_RECALL: [
                r'(?:recall|load|switch\s+to|go\s+to)\s+(?:scene\s+|preset\s+)?(.+)',
                r'(?:save|store)\s+(?:current\s+)?(?:mix|scene|preset)\s+(?:as\s+)?(.+)?',
            ],
            IntentType.QUERY_STATUS: [
                r'(?:what\'?s|what\s+is|show\s+me|tell\s+me)\s+(?:the\s+)?(?:current\s+)?(\w+)\s+(?:of|for|on)?\s+(?:channel\s+)?(\d+|vocals?|drums?|bass|guitar)?',
                r'(?:how\s+loud|what\s+level)\s+(?:is|are)\s+(?:the\s+)?(\d+|vocals?|drums?|bass|guitar)?',
            ],
        }
        
        # Channel name mappings following industry standards
        self.channel_mappings = {
            "vocals": 1, "vocal": 1, "voice": 1, "singer": 1, "lead": 1,
            "drums": 2, "drum": 2, "kit": 2, "percussion": 2, "kick": 2,
            "bass": 3, "baseline": 3, "low": 3,
            "guitar": 4, "guitars": 4, "gtr": 4, "rhythm": 4,
            "keys": 5, "keyboard": 5, "piano": 5, "synth": 5, "pad": 5,
            "strings": 6, "violin": 6, "orchestra": 6,
            "horns": 7, "brass": 7, "trumpet": 7, "sax": 7,
            "fx": 8, "effects": 8, "ambient": 8, "atmosphere": 8,
        }
        
        # Value mappings for natural language
        self.value_mappings = {
            "louder": "+6", "quieter": "-6", "softer": "-3",
            "up": "+3", "down": "-3", "way up": "+9", "way down": "-9",
            "boost": "+3", "cut": "-3", "boost more": "+6", "cut more": "-6",
            "left": "-100", "right": "100", "center": "0", "middle": "0",
            "full left": "-100", "full right": "100", "hard left": "-100", "hard right": "100",
            "maximum": "100", "minimum": "0", "max": "100", "min": "0",
            "half": "50", "quarter": "25", "three quarters": "75",
        }
        
        # Context management
        self.dialog_contexts: Dict[str, DialogContext] = {}
        
        # Performance tracking following Meta AI practices
        self.metrics = {
            "commands_processed": 0,
            "intent_accuracy": 0.0,
            "entity_accuracy": 0.0,
            "response_times": deque(maxlen=100),
            "intent_distribution": defaultdict(int),
        }
        
        # Learning system
        self.pattern_weights = defaultdict(lambda: 1.0)
        
        logger.info("Production NLP system initialized")
    
    async def process_command(self, 
                            text: str, 
                            session_id: str = "default") -> ParsedCommand:
        """
        Process voice command with intent classification and entity extraction
        Following Hugging Face pipeline pattern
        """
        start_time = datetime.now()
        
        try:
            # Get or create dialog context
            if session_id not in self.dialog_contexts:
                self.dialog_contexts[session_id] = DialogContext(session_id=session_id)
            context = self.dialog_contexts[session_id]
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Intent classification
            intent, intent_confidence = self._classify_intent(processed_text)
            
            # Entity extraction
            entities = self._extract_entities(processed_text, intent)
            
            # Apply context resolution
            entities = self._resolve_context(entities, context)
            
            # Create parsed command
            parsed = ParsedCommand(
                raw_text=text,
                intent=intent,
                intent_confidence=intent_confidence,
                entities=entities,
                timestamp=datetime.now(),
                session_id=session_id,
                context={"previous_command": context.last_command.raw_text if context.last_command else None}
            )
            
            # Update context
            context.last_command = parsed
            context.command_history.append(parsed)
            
            # Update channel context if found
            channel_entities = [e for e in entities if e.entity_type == EntityType.CHANNEL]
            if channel_entities:
                context.last_channel = channel_entities[0].value
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["commands_processed"] += 1
            self.metrics["response_times"].append(processing_time)
            self.metrics["intent_distribution"][intent] += 1
            
            return parsed
            
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return ParsedCommand(
                raw_text=text,
                intent=IntentType.UNKNOWN,
                intent_confidence=0.0,
                entities=[],
                timestamp=datetime.now(),
                session_id=session_id
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text following NLP best practices"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Expand contractions
        contractions = {
            "what's": "what is", "it's": "it is", "don't": "do not",
            "can't": "cannot", "won't": "will not", "i'm": "i am",
            "you're": "you are", "we're": "we are", "they're": "they are"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove filler words but preserve command structure
        filler_pattern = r'\b(um|uh|like|you know|please|thanks|okay|alright)\b'
        text = re.sub(filler_pattern, '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        Classify intent using pattern matching
        Following Meta AI classification approach
        """
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    match_coverage = len(match.group(0)) / len(text)
                    pattern_weight = self.pattern_weights[pattern]
                    
                    confidence = (match_coverage * 0.6 + pattern_weight * 0.4) * 0.9
                    
                    # Boost for exact matches
                    if match.group(0).strip() == text.strip():
                        confidence += 0.1
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        return best_intent, min(best_confidence, 1.0)
    
    def _extract_entities(self, text: str, intent: IntentType) -> List[Entity]:
        """
        Extract named entities based on intent
        Following structured entity extraction patterns
        """
        entities = []
        
        # Channel entity extraction
        channel_patterns = [
            r'\b(?:channel\s+)?(\d+)\b',
            r'\b(vocals?|drums?|bass|guitar|keys?|keyboard|piano|synth|strings?|horns?|brass|fx|effects)\b'
        ]
        
        for pattern in channel_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                channel_text = match.group(1)
                channel_num = self._resolve_channel(channel_text)
                if channel_num:
                    entities.append(Entity(
                        entity_type=EntityType.CHANNEL,
                        value=channel_num,
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        # Value entity extraction
        value_patterns = [
            r'\b([\+\-]?\d+(?:\.\d+)?)\b',  # Numbers
            r'\b(louder|quieter|up|down|left|right|center|middle|maximum|minimum|max|min|half|quarter)\b'  # Named values
        ]
        
        for pattern in value_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_text = match.group(1)
                resolved_value = self._resolve_value(value_text)
                if resolved_value:
                    entities.append(Entity(
                        entity_type=EntityType.VALUE,
                        value=resolved_value,
                        confidence=0.8,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        # Parameter entity extraction (for EQ, effects, etc.)
        if intent in [IntentType.EQ_ADJUST, IntentType.EFFECT_CONTROL]:
            param_patterns = [
                r'\b(bass|mid|treble|highs?|lows?|mids?|presence|air)\b',  # EQ bands
                r'\b(reverb|delay|chorus|compression|compressor|gate|limiter)\b'  # Effects
            ]
            
            for pattern in param_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    param_type = EntityType.PARAMETER if intent == IntentType.EQ_ADJUST else EntityType.EFFECT
                    entities.append(Entity(
                        entity_type=param_type,
                        value=match.group(1).lower(),
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        # Direction entity extraction
        direction_pattern = r'\b(left|right|center|middle)\b'
        matches = re.finditer(direction_pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append(Entity(
                entity_type=EntityType.DIRECTION,
                value=match.group(1).lower(),
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Scene entity extraction
        if intent == IntentType.SCENE_RECALL:
            scene_pattern = r'(?:scene|preset)\s+(.+?)(?:\s|$)'
            match = re.search(scene_pattern, text, re.IGNORECASE)
            if match:
                entities.append(Entity(
                    entity_type=EntityType.SCENE,
                    value=match.group(1).strip(),
                    confidence=0.8,
                    start_pos=match.start(1),
                    end_pos=match.end(1)
                ))
        
        return entities
    
    def _resolve_channel(self, channel_text: str) -> Optional[int]:
        """Resolve channel name or number to channel ID"""
        # Try direct number
        if channel_text.isdigit():
            channel_num = int(channel_text)
            return channel_num if 1 <= channel_num <= 16 else None
        
        # Try channel name mapping
        channel_lower = channel_text.lower()
        return self.channel_mappings.get(channel_lower)
    
    def _resolve_value(self, value_text: str) -> Optional[str]:
        """Resolve value text to numeric or symbolic value"""
        # Check value mappings first
        value_lower = value_text.lower()
        if value_lower in self.value_mappings:
            return self.value_mappings[value_lower]
        
        # Try to parse as number
        try:
            float_val = float(value_text)
            return str(float_val)
        except ValueError:
            pass
        
        return value_text
    
    def _resolve_context(self, entities: List[Entity], context: DialogContext) -> List[Entity]:
        """Resolve entities using dialog context"""
        # If no channel specified but we have a last channel, add it
        has_channel = any(e.entity_type == EntityType.CHANNEL for e in entities)
        
        if not has_channel and context.last_channel:
            entities.append(Entity(
                entity_type=EntityType.CHANNEL,
                value=context.last_channel,
                confidence=0.7,  # Lower confidence for inferred
                start_pos=-1,
                end_pos=-1
            ))
        
        return entities
    
    async def execute_command(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Execute parsed command
        Following production execution patterns
        """
        try:
            if parsed.intent == IntentType.UNKNOWN or parsed.intent_confidence < 0.5:
                return {
                    "success": False,
                    "error": "Command not understood",
                    "confidence": parsed.intent_confidence,
                    "suggestion": "Try rephrasing your command"
                }
            
            # Extract entities by type
            channels = [e.value for e in parsed.entities if e.entity_type == EntityType.CHANNEL]
            values = [e.value for e in parsed.entities if e.entity_type == EntityType.VALUE]
            parameters = [e.value for e in parsed.entities if e.entity_type == EntityType.PARAMETER]
            effects = [e.value for e in parsed.entities if e.entity_type == EntityType.EFFECT]
            directions = [e.value for e in parsed.entities if e.entity_type == EntityType.DIRECTION]
            scenes = [e.value for e in parsed.entities if e.entity_type == EntityType.SCENE]
            
            # Execute based on intent
            result = {"success": True, "intent": parsed.intent.value, "actions": []}
            
            if parsed.intent == IntentType.VOLUME_ADJUST:
                channel = channels[0] if channels else 1
                value = values[0] if values else "+3"
                result["actions"].append({
                    "type": "volume_adjust",
                    "channel": channel,
                    "value": value
                })
            
            elif parsed.intent == IntentType.PAN_ADJUST:
                channel = channels[0] if channels else 1
                direction = directions[0] if directions else "center"
                pan_value = {"left": -100, "right": 100, "center": 0, "middle": 0}.get(direction, 0)
                result["actions"].append({
                    "type": "pan_adjust", 
                    "channel": channel,
                    "value": pan_value
                })
            
            elif parsed.intent == IntentType.EQ_ADJUST:
                channel = channels[0] if channels else 1
                parameter = parameters[0] if parameters else "mid"
                value = values[0] if values else "+3"
                result["actions"].append({
                    "type": "eq_adjust",
                    "channel": channel,
                    "parameter": parameter,
                    "value": value
                })
            
            elif parsed.intent == IntentType.EFFECT_CONTROL:
                channel = channels[0] if channels else 1
                effect = effects[0] if effects else "reverb"
                # Check if it's add or remove based on original text
                action = "add" if any(word in parsed.raw_text.lower() for word in ["add", "apply", "enable", "turn on"]) else "remove"
                result["actions"].append({
                    "type": "effect_control",
                    "channel": channel,
                    "effect": effect,
                    "action": action
                })
            
            elif parsed.intent == IntentType.MUTE_TOGGLE:
                channel = channels[0] if channels else 1
                result["actions"].append({
                    "type": "mute_toggle",
                    "channel": channel
                })
            
            elif parsed.intent == IntentType.SOLO_TOGGLE:
                channel = channels[0] if channels else 1
                result["actions"].append({
                    "type": "solo_toggle",
                    "channel": channel
                })
            
            elif parsed.intent == IntentType.SCENE_RECALL:
                scene = scenes[0] if scenes else "default"
                result["actions"].append({
                    "type": "scene_recall",
                    "scene": scene
                })
            
            elif parsed.intent == IntentType.QUERY_STATUS:
                channel = channels[0] if channels else 1
                parameter = parameters[0] if parameters else "volume"
                result["actions"].append({
                    "type": "query_status",
                    "channel": channel,
                    "parameter": parameter
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "intent": parsed.intent.value
            }
    
    def get_suggestions(self, partial_text: str) -> List[str]:
        """Get command suggestions for autocomplete"""
        suggestions = []
        partial_lower = partial_text.lower()
        
        # Command templates based on intents
        templates = [
            "set volume of {channel} to {value}",
            "pan {channel} to {direction}",
            "boost the {parameter} on {channel}",
            "add {effect} to {channel}",
            "mute {channel}",
            "solo {channel}",
            "recall scene {scene}",
            "what is the volume of {channel}",
        ]
        
        # Expand templates with common values
        channels = ["vocals", "drums", "bass", "guitar", "channel 1", "channel 2"]
        values = ["50", "75", "100", "louder", "quieter"]
        directions = ["left", "right", "center"]
        parameters = ["bass", "mid", "treble"]
        effects = ["reverb", "delay", "compression"]
        scenes = ["live", "studio", "broadcast"]
        
        for template in templates:
            if template.startswith(partial_lower):
                # Fill in template with examples
                filled = template.format(
                    channel="vocals",
                    value="75",
                    direction="left",
                    parameter="bass",
                    effect="reverb",
                    scene="live"
                )
                suggestions.append(filled)
        
        return suggestions[:5]
    
    def train_from_feedback(self, parsed: ParsedCommand, success: bool):
        """Update system based on execution feedback"""
        # Find the pattern that matched this command
        intent_patterns = self.intent_patterns.get(parsed.intent, [])
        for pattern in intent_patterns:
            if re.search(pattern, parsed.raw_text, re.IGNORECASE):
                if success:
                    self.pattern_weights[pattern] *= 1.1  # Boost successful patterns
                else:
                    self.pattern_weights[pattern] *= 0.9  # Reduce failed patterns
                break
        
        # Update accuracy metrics
        if success:
            self.metrics["intent_accuracy"] = (self.metrics["intent_accuracy"] * 0.9 + 1.0 * 0.1)
            self.metrics["entity_accuracy"] = (self.metrics["entity_accuracy"] * 0.9 + 1.0 * 0.1)
        else:
            self.metrics["intent_accuracy"] *= 0.95
            self.metrics["entity_accuracy"] *= 0.95
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        avg_response_time = np.mean(self.metrics["response_times"]) if self.metrics["response_times"] else 0
        
        return {
            "commands_processed": self.metrics["commands_processed"],
            "intent_accuracy": f"{self.metrics['intent_accuracy']:.1%}",
            "entity_accuracy": f"{self.metrics['entity_accuracy']:.1%}",
            "average_response_time_ms": f"{avg_response_time:.1f}",
            "active_sessions": len(self.dialog_contexts),
            "intent_distribution": dict(self.metrics["intent_distribution"]),
            "system_status": "healthy" if avg_response_time < 100 else "degraded"
        }


async def demo_production_nlp():
    """Demo the production NLP system"""
    print("🎤 Production NLP System Demo")
    print("=" * 50)
    
    # Initialize system
    nlp = ProductionNLP()
    print("✅ Production NLP system initialized")
    
    # Test commands following real-world usage patterns
    test_commands = [
        "set the volume of vocals to 75",
        "pan the drums to the left",
        "boost the bass on channel 3",
        "add reverb to vocals",
        "mute channel 2",
        "solo the guitar",
        "recall the live scene",
        "what is the volume of the bass",
        "turn up the vocals",
        "make it louder",  # Context test
    ]
    
    print("\n🔄 Processing test commands...")
    print("-" * 40)
    
    session_id = "demo_session"
    
    for i, command_text in enumerate(test_commands, 1):
        print(f"\n{i:2d}. Input: '{command_text}'")
        
        # Process command
        parsed = await nlp.process_command(command_text, session_id)
        
        print(f"    Intent: {parsed.intent.value} ({parsed.intent_confidence:.1%})")
        
        if parsed.entities:
            print(f"    Entities:")
            for entity in parsed.entities:
                print(f"      {entity.entity_type.value}: {entity.value} ({entity.confidence:.1%})")
        
        # Execute command
        result = await nlp.execute_command(parsed)
        
        if result["success"]:
            print(f"    ✅ Actions: {len(result['actions'])}")
            for action in result["actions"]:
                print(f"       → {action['type']}: {action}")
            
            # Provide feedback for learning
            nlp.train_from_feedback(parsed, True)
        else:
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
            nlp.train_from_feedback(parsed, False)
    
    # Test autocomplete
    print("\n\n💡 Autocomplete Demo:")
    print("-" * 40)
    
    partial_commands = ["set vol", "pan", "boost", "add rev"]
    for partial in partial_commands:
        suggestions = nlp.get_suggestions(partial)
        print(f"\n'{partial}...' →")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
    
    # Show performance metrics
    print("\n\n📊 Performance Metrics:")
    print("-" * 40)
    
    metrics = nlp.get_performance_metrics()
    for key, value in metrics.items():
        if key != "intent_distribution":
            print(f"{key}: {value}")
    
    print("\nIntent Distribution:")
    for intent, count in metrics["intent_distribution"].items():
        print(f"  {intent}: {count}")
    
    print("\n✅ Production NLP Demo Complete")
    return True


if __name__ == "__main__":
    # Test the production system
    success = asyncio.run(demo_production_nlp())
    print(f"\nDemo success: {success}")