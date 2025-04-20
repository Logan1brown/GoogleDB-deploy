"""User messaging system for the data entry app."""
from enum import Enum
import streamlit as st


class MessageType(Enum):
    """Types of messages that can be displayed to users."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class MessageCategory(Enum):
    """Categories of messages for different parts of the application."""
    VALIDATION = "validation"
    AUTH = "auth"
    DATA_ENTRY = "data_entry"
    DATABASE = "database"


class UserMessage:
    """Handle consistent user messaging throughout the application."""
    
    @staticmethod
    def show(message: str, msg_type: MessageType, category: MessageCategory) -> None:
        """Display a message to the user with consistent styling.
        
        Args:
            message: The message to display
            msg_type: Type of message (error, warning, info, success)
            category: Category the message belongs to
        """
        prefix = {
            MessageCategory.VALIDATION: "🔍 Validation",
            MessageCategory.AUTH: "🔐 Authentication",
            MessageCategory.DATA_ENTRY: "📝 Data Entry",
            MessageCategory.DATABASE: "💾 Database"
        }
        
        formatted_msg = f"{prefix[category]}: {message}"
        
        if msg_type == MessageType.ERROR:
            st.error(formatted_msg)
        elif msg_type == MessageType.WARNING:
            st.warning(formatted_msg)
        elif msg_type == MessageType.INFO:
            st.info(formatted_msg)
        elif msg_type == MessageType.SUCCESS:
            st.success(formatted_msg)


# Common validation messages
SAME_GENRE_SUBGENRE = "Genre and Subgenre must be different"
MISSING_REQUIRED = "Required field: {field}"
INVALID_FORMAT = "Invalid format for {field}"
