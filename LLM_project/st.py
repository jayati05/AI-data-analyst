"""Enhanced Streamlit application for data analysis with superior UX."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from logging_client import log_debug, log_error, log_info, log_success, log_warning

# Constants
API_URL = "http://localhost:8000"
DATA_FOLDER = Path("./data")
PLOTS_FOLDER = Path("./plots")
TIMEOUT = 30  # seconds
HTTP_OK = 200
SUPPORTED_PLOT_FORMATS = (".png", ".jpg", ".jpeg", ".svg")
MAX_PLOTS_TO_KEEP = 20
MAX_PLOTS_TO_DISPLAY = 5
MAX_FILE_SIZE_MB = 10  # Maximum file size to accept in MB

# Initialize logging
log_info("Initializing Enhanced Streamlit application")

# Create necessary directories if they don't exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

def convert_bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return size_bytes / (1024 * 1024)

def attempt_plot_deletions(
    plots: list[Path],
) -> tuple[int, list[tuple[Path, Exception]]]:
    """Attempt to delete plots and return success count and failures."""
    success_count = 0
    failures = []

    for plot in plots:
        if not plot.exists():
            failures.append((plot, FileNotFoundError(f"File not found: {plot}")))
            continue

        try:
            plot.unlink()
            success_count += 1
        except OSError as e:
            failures.append((plot, e))

    return success_count, failures

def retry_failed_deletions(
    failures: list[tuple[Path, Exception]],
) -> tuple[int, list[tuple[Path, Exception]]]:
    """Retry failed deletions and return results."""
    retry_success = 0
    permanent_failures = []

    for plot, _ in failures:
        if not plot.exists():
            retry_success += 1  # Already gone, count as success
            continue

        try:
            plot.unlink()
            retry_success += 1
        except OSError as e:
            permanent_failures.append((plot, e))

    return retry_success, permanent_failures

def cleanup_plots(max_plots: int = MAX_PLOTS_TO_KEEP) -> None:
    """Keep only the most recent plots to avoid clutter."""
    try:
        plot_files = sorted(PLOTS_FOLDER.iterdir(), key=os.path.getmtime, reverse=True)
        files_to_delete = plot_files[max_plots:]

        success_count, failures = attempt_plot_deletions(files_to_delete)

        if failures:
            log_warning(f"Initial deletion failed for {len(failures)} plots")
            retry_success, permanent_failures = retry_failed_deletions(failures)

            if permanent_failures:
                for plot, error in permanent_failures:
                    log_error(f"Permanent failure deleting {plot.name}: {error}")

            deleted = success_count + retry_success
            log_info(f"Deleted {deleted}/{len(files_to_delete)} plots")
        else:
            log_info(f"Successfully deleted {success_count} plots")

    except OSError as e:
        log_error(f"Plot directory access failed: {e}")

cleanup_plots()

# Page setup
try:
    st.set_page_config(
        page_title="Data Analysis Chat",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
    )
    st.title("📊 Data Analysis Chat")
    st.caption("Analyze your data with natural language queries")
    log_debug("Page configuration set")
except Exception as e:
    log_error(f"Page setup failed: {e!s}")
    raise

def inject_custom_css() -> None:
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
        .stChatMessage {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .stChatMessage.user {
            background-color: #f0f2f6;
        }
        .stChatMessage.assistant {
            background-color: #ffffff;
            border: 1px solid #e1e4e8;
        }
        .plot-expander {
            margin-top: 12px;
        }
        .file-uploader {
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        .stButton>button {
            border-radius: 4px;
            padding: 0.25rem 0.75rem;
        }
        .stDownloadButton>button {
            width: 100%;
            justify-content: center;
        }
        .plot-thumbnail {
            max-width: 100%;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .plot-thumbnail:hover {
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

if "messages" not in st.session_state:
    log_info("Initializing new chat session")
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your data analysis assistant. You can:\n\n"
                      "1. Upload data files (CSV/Excel) in the sidebar\n"
                      "2. Ask questions about your data\n"
                      "3. View generated plots and analysis steps\n\n"
                      "What would you like to analyze today?",
        },
    ]
    log_debug(f"Initial messages: {st.session_state.messages}")

def save_plot_from_response(response_data: dict[str, Any]) -> Path | None:
    """Save plot from API response and return its path."""
    try:
        if isinstance(response_data, dict) and "plot" in response_data:
            plot_data = response_data["plot"]
            plot_format = response_data.get("plot_format", ".png")

            if not plot_format.startswith("."):
                plot_format = f".{plot_format}"

            if plot_format not in SUPPORTED_PLOT_FORMATS:
                plot_format = ".png"

            timestamp = int(time.time())
            plot_name = f"plot_{timestamp}{plot_format}"
            plot_path = PLOTS_FOLDER / plot_name

            if plot_data.startswith("data:"):
                header, plot_data = plot_data.split(",", 1)
                plot_bytes = base64.b64decode(plot_data)
                with plot_path.open("wb") as f:
                    f.write(plot_bytes)
            else:
                plot_bytes = base64.b64decode(plot_data)
                with plot_path.open("wb") as f:
                    f.write(plot_bytes)

            log_success(f"Saved plot: {plot_name}")
            return plot_path

        if isinstance(response_data, dict) and "plot_file" in response_data:
            plot_file = response_data["plot_file"]
            plot_name = Path(plot_file).name
            plot_path = PLOTS_FOLDER / plot_name

            if "plot_content" in response_data:
                with plot_path.open("wb") as f:
                    f.write(base64.b64decode(response_data["plot_content"]))
            elif not plot_path.exists():
                log_warning(f"Plot file not found: {plot_path}")
                return None

            return plot_path

    except (OSError, ValueError, TypeError) as e:
        log_error(f"Failed to save plot: {e}")
        return None

def display_plot(plot_path: Path) -> None:
    """Display a plot with download and expand options."""
    try:
        if not plot_path.exists():
            st.warning(f"Plot file not found at: {plot_path}")
            log_warning(f"Plot not found: {plot_path}")
            return

        col1, col2 = st.columns([4, 1])

        with col1:
            if plot_path.suffix == ".svg":
                with plot_path.open() as f:
                    svg = f.read()
                st.image(svg, use_column_width=True)
            else:
                st.image(str(plot_path), use_column_width=True)

        with col2:
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
            }
            mime_type = mime_types.get(
                plot_path.suffix.lower(),
                "application/octet-stream",
            )

            with plot_path.open("rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=plot_path.name,
                    mime=mime_type,
                    key=f"dl_{plot_path.name}",
                    use_container_width=True,
                )

        log_success(f"Displayed plot: {plot_path.name}")

    except (OSError, ValueError) as e:
        st.error(f"Failed to display plot: {e!s}")
        log_error(f"Plot display failed: {e!s}")

def display_message(message: dict[str, Any]) -> None:
    """Display a chat message with logging and optional plot."""
    try:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            log_debug(f"Displayed {message['role']} message")

            if message.get("plot_path"):
                plot_path = Path(message["plot_path"])
                with st.expander("📊 View Plot", expanded=True):
                    display_plot(plot_path)

            if message.get("steps"):
                with st.expander("🔍 See analysis steps", expanded=False):
                    for i, step in enumerate(message["steps"], 1):
                        st.markdown(f"**Step {i}**")
                        st.code(step, language="python")
                    log_info("Displayed analysis steps")

    except (KeyError, AttributeError) as e:
        log_error(f"Failed to display message: {e!s}")
        st.error(f"Error displaying message: {e!s}")

def handle_file_upload() -> None:
    """Handle file uploads with validation and feedback."""
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV/Excel files (Max 10MB each)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload your data files for analysis",
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        file_size_mb = convert_bytes_to_mb(uploaded_file.size)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.sidebar.error(
                f"File {uploaded_file.name} is too large "
                f"({file_size_mb:.2f}MB). Max size is {MAX_FILE_SIZE_MB}MB.",
            )
            continue

        file_path = DATA_FOLDER / uploaded_file.name
        try:
            with st.spinner(f"Saving {uploaded_file.name}..."), \
                 file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(
                f"Saved: {uploaded_file.name} ({file_size_mb:.2f}MB)",
            )
            log_success(f"File uploaded: {uploaded_file.name}")
        except OSError as e:
            st.sidebar.error(f"Failed to save {uploaded_file.name}: {e!s}")
            log_error(f"File upload failed: {uploaded_file.name} - {e}")

def display_file_management() -> None:
    """Display file management interface in sidebar."""
    st.sidebar.header("📂 File Management")
    handle_file_upload()

    available_files = [
        f.name for f in DATA_FOLDER.iterdir()
        if f.suffix in (".csv", ".xlsx")
    ]
    if not available_files:
        st.sidebar.info("No data files found. Upload files to get started.")
        log_debug("No data files found in directory")
        return

    st.sidebar.subheader("Your Files")
    st.sidebar.caption("Click 🗑️ to delete a file")

    for file in available_files:
        cols = st.sidebar.columns([4, 1])
        cols[0].markdown(f"📄 {file}")

        if not cols[1].button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
            continue

        try:
            file_path = DATA_FOLDER / file
            file_path.unlink()
            st.sidebar.success(f"Deleted: {file}")
            log_info(f"File deleted: {file}")
            time.sleep(0.5)
            st.rerun()
        except OSError as e:
            st.sidebar.error(f"Failed to delete {file}: {e!s}")
            log_error(f"File deletion failed: {file} - {e}")

def display_plot_thumbnail(plot_file: Path) -> None:
    """Display a single plot thumbnail."""
    cols = st.sidebar.columns([3, 1])
    with cols[0]:
        if plot_file.suffix == ".svg":
            with plot_file.open() as f:
                svg = f.read()
            st.image(svg, use_column_width=True)
        else:
            st.image(str(plot_file), use_column_width=True)

    with cols[1]:
        if st.button(
            "👀",
            key=f"view_{plot_file.name}",
            help=f"View {plot_file.name}",
        ):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Showing plot: {plot_file.name}",
                "plot_path": str(plot_file),
            })
            st.rerun()

def handle_plot_deletion() -> None:
    """Handle the 'Clear All' plots functionality."""
    try:
        plot_files = list(PLOTS_FOLDER.glob("*"))
        success_count, failures = attempt_plot_deletions(plot_files)

        if failures:
            log_warning(f"Failed to delete {len(failures)} plots")
            for file, error in failures:
                log_error(f"Couldn't delete {file.name}: {error}")

        st.sidebar.success(f"Cleared {success_count} plots")
        time.sleep(0.5)
        st.rerun()
    except OSError as e:
        st.sidebar.error(f"Failed to access plots: {e!s}")

def display_plot_thumbnails(plot_files: list[Path]) -> None:
    """Display plot thumbnails with error handling."""
    display_errors = []

    for plot_file in plot_files[:MAX_PLOTS_TO_DISPLAY]:
        # Pre-check for common failure cases
        if not plot_file.exists():
            error = FileNotFoundError(f"File not found: {plot_file}")
            display_errors.append((plot_file, error))
            continue

        if plot_file.stat().st_size == 0:
            error = ValueError(f"Empty file: {plot_file}")
            display_errors.append((plot_file, error))
            continue

        try:
            display_plot_thumbnail(plot_file)
        except (OSError, ValueError) as e:
            display_errors.append((plot_file, e))

    for file, error in display_errors:
        st.sidebar.error(f"Couldn't display {file.name}")
        log_error(f"Thumbnail display failed for {file}: {error}")

def display_plot_management() -> None:
    """Display plot management interface in sidebar."""
    st.sidebar.header("📊 Plot Management")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🔄 Refresh", help="Refresh the plot list"):
            st.rerun()
    with col2:
        if st.sidebar.button("🧹 Clear All", help="Delete all generated plots"):
            handle_plot_deletion()

    plot_files = sorted(PLOTS_FOLDER.iterdir(), key=os.path.getmtime, reverse=True)
    if not plot_files:
        st.sidebar.info("No plots generated yet. Ask a question to generate plots.")
        return

    st.sidebar.subheader(f"Recent Plots (Last {MAX_PLOTS_TO_DISPLAY})")
    display_plot_thumbnails(plot_files)

def handle_api_response(
    response: requests.Response,
) -> tuple[str, list[str], Path | None]:
    """Handle API response and return content, steps, and plot path."""
    if response.status_code != HTTP_OK:
        error_msg = f"API Error {response.status_code}: {response.text[:200]}..."
        log_error(f"API request failed: {error_msg}")
        return error_msg, [], None

    try:
        result = response.json()
    except ValueError:
        return response.text, [], None

    full_response = result.get("output", "No response")
    steps = result.get("steps", [])
    plot_path = save_plot_from_response(result)
    return full_response, steps, plot_path

def process_user_query(prompt: str) -> None:
    """Process user query and display response."""
    try:
        log_info(f"New user input received: {prompt[:50]}...")
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})

        available_files = [
            f.name for f in DATA_FOLDER.iterdir()
            if f.suffix in (".csv", ".xlsx")
        ]
        file_context = (
            f"Available files: {', '.join(available_files)}"
            if available_files
            else "No files available"
        )
        full_prompt = f"{file_context}\n\nQuestion: {prompt}"
        request_data = {
            "query": full_prompt,
            "session_id": f"streamlit_session_{time.time()}",
            "plot_format": "png",
        }

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            full_response = ""
            steps = []
            plot_path = None

            try:
                with status_placeholder.status(
                    "Analyzing your data...",
                    expanded=True,
                ) as status:
                    st.write("🔍 Understanding your question...")
                    time.sleep(0.5)

                    log_info("Making API request to /execute-query")
                    start_time = time.time()
                    st.write("📡 Connecting to analysis engine...")

                    response = requests.post(
                        f"{API_URL}/execute-query",
                        json=request_data,
                        timeout=TIMEOUT,
                    )
                    processing_time = time.time() - start_time

                    full_response, steps, plot_path = handle_api_response(response)
                    message_placeholder.markdown(full_response)

                    status.update(
                        label="Analysis complete!",
                        state="complete",
                        expanded=False,
                    )
                    log_success(f"Query processed in {processing_time:.2f}s")

            except requests.Timeout:
                error_msg = "Request timed out. Please try again."
                status_placeholder.error(error_msg)
                full_response = error_msg
            except requests.ConnectionError:
                error_msg = "Could not connect to analysis service"
                status_placeholder.error(error_msg)
                full_response = error_msg
            except requests.RequestException as e:
                error_msg = f"Unexpected error: {e!s}"
                status_placeholder.error(error_msg)
                full_response = error_msg

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "steps": steps,
                "plot_path": str(plot_path) if plot_path else None,
            })

    except (OSError) as e:
        log_error(f"Chat input processing failed: {e!s}")
        st.error(f"System error: {e!s}")

def main() -> None:
    """Run the main application."""
    with st.sidebar:
        display_file_management()
        st.sidebar.divider()
        display_plot_management()

    st.subheader("Chat with your Data")

    for message in st.session_state.messages:
        display_message(message)

    if prompt := st.chat_input("Ask about your data...", key="chat_input"):
        process_user_query(prompt)

if __name__ == "__main__":
    main()
    log_debug("Streamlit application running")
