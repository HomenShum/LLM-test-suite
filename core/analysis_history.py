"""
Analysis History Manager

Saves and loads Test 6 analysis results to/from local storage.
Allows users to view previous analysis runs.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil


class AnalysisHistoryManager:
    """Manages saving and loading of analysis results."""
    
    def __init__(self, history_dir: str = "analysis_history"):
        """
        Initialize the history manager.
        
        Args:
            history_dir: Directory to store analysis history
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.results_dir = self.history_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.history_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.history_dir / "metadata.json"
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file or create new one."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"analyses": []}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_analysis(
        self,
        results: List[Dict[str, Any]],
        preset_name: str,
        task_description: str,
        selected_models: List[str],
        ground_truths: Optional[Dict[str, Any]] = None,
        curation_report: Optional[Dict[str, Any]] = None,
        computational_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        qa_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Save an analysis run to local storage.
        
        Args:
            results: Analysis results
            preset_name: Name of preset
            task_description: Task description
            selected_models: List of models used
            ground_truths: Ground truth data
            curation_report: Curation report
            computational_results: Computational analysis results
            evaluation_results: Model evaluation results
            qa_history: Q&A conversation history
        
        Returns:
            Analysis ID (timestamp-based)
        """
        # Generate unique ID
        timestamp = datetime.now()
        analysis_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Create analysis directory
        analysis_dir = self.results_dir / analysis_id
        analysis_dir.mkdir(exist_ok=True)
        
        # Copy images to history
        image_dir = self.images_dir / analysis_id
        image_dir.mkdir(exist_ok=True)
        
        copied_images = []
        for result in results:
            image_path = Path(result.get('image_path', ''))
            if image_path.exists():
                dest_path = image_dir / image_path.name
                shutil.copy2(image_path, dest_path)
                copied_images.append(str(dest_path))
        
        # Prepare serializable results (convert Pydantic models to dicts)
        serializable_results = []
        for result in results:
            serializable_result = {
                "image_path": result.get('image_path', ''),
                "image_name": result.get('image_name', ''),
                "model_results": {}
            }
            
            for model_name, analysis in result.get("model_results", {}).items():
                # Convert Pydantic model to dict
                if hasattr(analysis, 'model_dump'):
                    serializable_result["model_results"][model_name] = analysis.model_dump()
                elif hasattr(analysis, 'dict'):
                    serializable_result["model_results"][model_name] = analysis.dict()
                else:
                    serializable_result["model_results"][model_name] = str(analysis)
            
            serializable_results.append(serializable_result)
        
        # Save results as JSON
        results_file = analysis_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": serializable_results,
                "preset_name": preset_name,
                "task_description": task_description,
                "selected_models": selected_models,
                "timestamp": timestamp.isoformat(),
                "num_images": len(results),
                "num_models": len(selected_models)
            }, f, indent=2)
        
        # Save ground truths if available
        if ground_truths:
            ground_truth_file = analysis_dir / "ground_truths.json"
            with open(ground_truth_file, 'w', encoding='utf-8') as f:
                json.dump(ground_truths, f, indent=2)
        
        # Save curation report if available
        if curation_report:
            curation_file = analysis_dir / "curation_report.json"
            with open(curation_file, 'w', encoding='utf-8') as f:
                json.dump(curation_report, f, indent=2)
        
        # Save computational results if available
        if computational_results:
            comp_file = analysis_dir / "computational_results.json"
            with open(comp_file, 'w', encoding='utf-8') as f:
                json.dump(computational_results, f, indent=2)
        
        # Save evaluation results if available
        if evaluation_results:
            eval_file = analysis_dir / "evaluation_results.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2)
        
        # Save Q&A history if available
        if qa_history:
            qa_file = analysis_dir / "qa_history.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_history, f, indent=2)
        
        # Update metadata
        self.metadata["analyses"].append({
            "id": analysis_id,
            "timestamp": timestamp.isoformat(),
            "preset_name": preset_name,
            "task_description": task_description,
            "num_images": len(results),
            "num_models": len(selected_models),
            "models": selected_models,
            "has_ground_truth": ground_truths is not None,
            "has_curation_report": curation_report is not None,
            "has_computational_results": computational_results is not None,
            "has_evaluation_results": evaluation_results is not None,
            "has_qa_history": qa_history is not None
        })
        
        self._save_metadata()
        
        return analysis_id
    
    def load_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an analysis by ID.
        
        Args:
            analysis_id: Analysis ID to load
        
        Returns:
            Dict with all analysis data, or None if not found
        """
        analysis_dir = self.results_dir / analysis_id
        
        if not analysis_dir.exists():
            return None
        
        # Load main results
        results_file = analysis_dir / "results.json"
        if not results_file.exists():
            return None
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load ground truths if available
        ground_truth_file = analysis_dir / "ground_truths.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                data["ground_truths"] = json.load(f)
        
        # Load curation report if available
        curation_file = analysis_dir / "curation_report.json"
        if curation_file.exists():
            with open(curation_file, 'r', encoding='utf-8') as f:
                data["curation_report"] = json.load(f)
        
        # Load computational results if available
        comp_file = analysis_dir / "computational_results.json"
        if comp_file.exists():
            with open(comp_file, 'r', encoding='utf-8') as f:
                data["computational_results"] = json.load(f)
        
        # Load evaluation results if available
        eval_file = analysis_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                data["evaluation_results"] = json.load(f)
        
        # Load Q&A history if available
        qa_file = analysis_dir / "qa_history.json"
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                data["qa_history"] = json.load(f)
        
        # Update image paths to point to saved copies
        image_dir = self.images_dir / analysis_id
        if image_dir.exists():
            for result in data.get("results", []):
                image_name = Path(result.get("image_path", "")).name
                saved_image_path = image_dir / image_name
                if saved_image_path.exists():
                    result["image_path"] = str(saved_image_path)
        
        return data
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """
        Get list of all saved analyses.
        
        Returns:
            List of analysis metadata dicts
        """
        # Sort by timestamp (newest first)
        analyses = sorted(
            self.metadata.get("analyses", []),
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return analyses
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis by ID.
        
        Args:
            analysis_id: Analysis ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        analysis_dir = self.results_dir / analysis_id
        image_dir = self.images_dir / analysis_id
        
        # Delete directories
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
        
        if image_dir.exists():
            shutil.rmtree(image_dir)
        
        # Remove from metadata
        self.metadata["analyses"] = [
            a for a in self.metadata.get("analyses", [])
            if a.get("id") != analysis_id
        ]
        
        self._save_metadata()
        
        return True
    
    def get_analysis_summary(self, analysis_id: str) -> Optional[str]:
        """
        Get a human-readable summary of an analysis.
        
        Args:
            analysis_id: Analysis ID
        
        Returns:
            Summary string
        """
        for analysis in self.metadata.get("analyses", []):
            if analysis.get("id") == analysis_id:
                timestamp = datetime.fromisoformat(analysis.get("timestamp", ""))
                return (
                    f"{analysis.get('preset_name', 'Unknown')} - "
                    f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"{analysis.get('num_images', 0)} images, "
                    f"{analysis.get('num_models', 0)} models"
                )
        return None

