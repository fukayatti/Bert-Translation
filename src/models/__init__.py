"""
モデルモジュール

TeacherモデルとStudentモデルの定義と管理を行います。
"""

from .teacher_model import TeacherModel
from .student_model import StudentModel

__all__ = ["TeacherModel", "StudentModel"]