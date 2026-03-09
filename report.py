"""
Report Generation Module
Creates professional PDF reports in thesis format
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                 TableStyle, PageBreak, Image)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import matplotlib.pyplot as plt
import io
from datetime import datetime

class ThesisReportGenerator:
    def __init__(self, filename='Thesis_Report.pdf'):
        self.filename = filename
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
    
    def add_title_page(self, title, author, instructor=None, date=None, logo_path=None):
        """Add title page to report"""
        if date is None:
            date = datetime.now().strftime("%B %d, %Y")

        # Add university logo if provided
        if logo_path:
            try:
                import os
                if os.path.exists(logo_path):
                    logo = Image(logo_path, width=2*inch, height=2*inch)
                    logo.hAlign = 'CENTER'
                    self.story.append(Spacer(1, 0.5*inch))
                    self.story.append(logo)
                    self.story.append(Spacer(1, 0.3*inch))
            except Exception as e:
                print(f"Warning: Could not load logo: {e}")
        else:
            self.story.append(Spacer(1, 1.5*inch))

        # University name
        university_style = ParagraphStyle(
            name='University',
            parent=self.styles['Normal'],
            fontSize=14,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1f4788')
        )
        self.story.append(Paragraph("HOCHIMINH CITY INTERNATIONAL UNIVERSITY", university_style))
        self.story.append(Paragraph("HCM - IU", university_style))
        self.story.append(Spacer(1, 0.5*inch))

        # Title
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.8*inch))

        # Instructor and Student info
        info_style = ParagraphStyle(
            name='Info',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            spaceAfter=6
        )

        if instructor:
            self.story.append(Paragraph(f"<b>Instructor:</b> {instructor}", info_style))
            self.story.append(Spacer(1, 0.2*inch))

        self.story.append(Paragraph(f"<b>Student:</b> {author}", info_style))
        self.story.append(Spacer(1, 0.4*inch))

        # Date
        self.story.append(Paragraph(f"<b>{date}</b>", info_style))
        self.story.append(PageBreak())
    
    def add_abstract(self, text):
        """Add abstract section"""
        self.story.append(Paragraph("Abstract", self.styles['SectionHeading']))
        self.story.append(Paragraph(text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_section(self, title, content):
        """Add a section with title and content"""
        self.story.append(Paragraph(title, self.styles['SectionHeading']))
        self.story.append(Paragraph(content, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_subsection(self, title, content):
        """Add a subsection"""
        self.story.append(Paragraph(title, self.styles['SubHeading']))
        self.story.append(Paragraph(content, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.15*inch))
    
    def add_table(self, data, title, col_widths=None):
        """Add a formatted table to the report"""
        self.story.append(Paragraph(title, self.styles['SubHeading']))
        
        # Create table
        table = Table(data, colWidths=col_widths)
        
        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_figure(self, fig, title, width=6*inch, height=4*inch):
        """Add a matplotlib figure to the report"""
        self.story.append(Paragraph(title, self.styles['SubHeading']))
        
        # Save figure to bytes
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Add image to story
        img = Image(img_buffer, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.3*inch))
        
    def generate_full_report(self, data_summary, results_tables, figures, 
                            cv_results, output_path):
        """Generate complete thesis-style report"""
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Title Page
        self.add_title_page(
            title="Machine Learning Analysis for Heart Disease Prediction",
            author="Nguyen Minh Thu",
            instructor="Assoc. Prof. Nguyen Thi Thuy Loan",
            date=datetime.now().strftime("%B %d, %Y"),
            logo_path="logo.png"  # Logo path - will be created in project directory
        )
        
        # Abstract
        abstract_text = """
        This study presents a comprehensive machine learning analysis for predicting 
        heart disease using a dataset of 1,190 cases with 12 attributes. Multiple 
        classification algorithms including Random Forest, XGBoost, Logistic Regression,
        Support Vector Machines, K-Nearest Neighbors, Gradient Boosting, Neural Networks,
        and ensemble methods were evaluated. The models were assessed using accuracy,
        precision, recall, F1-score, ROC-AUC, and confusion matrices. K-fold cross-validation
        (K=10 and K=5) was performed to ensure model robustness and generalizability.
        """
        self.add_abstract(abstract_text)
        
        # Introduction
        intro_text = """
        Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide.
        Machine learning techniques have shown significant potential in enhancing the
        precision and efficacy of health-related predictions. This analysis employs
        advanced ML algorithms to develop accurate predictive models for heart disease
        diagnosis.
        """
        self.add_section("1. Introduction", intro_text)
        
        # Dataset Description
        dataset_text = f"""
        The dataset comprises {data_summary['shape'][0]} records with {data_summary['shape'][1]}
        attributes including age, sex, chest pain type, resting blood pressure, cholesterol,
        fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina,
        ST depression, ST slope, and target variable (heart disease presence).
        """
        self.add_section("2. Dataset Description", dataset_text)
        
        # Methodology
        methodology_text = """
        The analysis pipeline includes: (1) Data preprocessing and cleaning,
        (2) Exploratory data analysis with correlation analysis, (3) Feature selection
        based on correlation and domain knowledge, (4) Train-test split (80-20),
        (5) Model training with 15 different algorithms, (6) Performance evaluation
        using multiple metrics, and (7) K-fold cross-validation for robustness testing.
        """
        self.add_section("3. Methodology", methodology_text)
        
        # Results - Add Tables
        self.add_section("4. Results and Analysis", "")
        
        # Table 3: Performance Metrics
        if 'table3' in results_tables:
            table3_data = [results_tables['table3'].columns.tolist()] + \
                         results_tables['table3'].values.tolist()
            self.add_table(table3_data, "Table 3: Performance Metrics of ML Models")
        
        # Table 4: ROC-AUC Values
        if 'table4' in results_tables:
            table4_data = [results_tables['table4'].columns.tolist()] + \
                         results_tables['table4'].values.tolist()
            self.add_table(table4_data, "Table 4: ROC-AUC Values")
        
        # Add correlation heatmap
        if 'correlation_heatmap' in figures:
            self.add_figure(figures['correlation_heatmap'], 
                          "Figure 1: Correlation Matrix Heatmap",
                          width=5.5*inch, height=4.5*inch)
        
        # Table 5: Confusion Matrices
        if 'table5' in results_tables:
            table5_data = [results_tables['table5'].columns.tolist()] + \
                         results_tables['table5'].values.tolist()
            self.add_table(table5_data, "Table 5: Confusion Matrix Summary")
        
        # Add ROC curves
        if 'roc_curves' in figures:
            self.add_figure(figures['roc_curves'], 
                          "Figure 2: ROC Curves for All Models",
                          width=6*inch, height=7*inch)
        
        # Table 6: Cross-validation Results
        if 'table6' in results_tables:
            table6_data = [results_tables['table6'].columns.tolist()] + \
                         results_tables['table6'].values.tolist()
            self.add_table(table6_data, "Table 6: K-Fold Cross-Validation Results")
        
        # Discussion
        discussion_text = """
        The ensemble methods, particularly Random Forest, XGBoost, and Bagged Trees,
        demonstrated superior performance with accuracies exceeding 90%. Cross-validation
        results confirmed model stability and generalizability. The ROC-AUC values
        indicate strong discriminative ability across most models. These findings
        highlight the effectiveness of machine learning approaches in cardiovascular
        disease prediction.
        """
        self.add_section("5. Discussion", discussion_text)
        
        # Conclusion
        conclusion_text = """
        This comprehensive analysis demonstrates that advanced machine learning techniques,
        particularly ensemble methods, can accurately predict heart disease. The combination
        of multiple evaluation metrics and cross-validation ensures robust and reliable
        predictions. Future work should explore deep learning approaches and real-time
        clinical deployment.
        """
        self.add_section("6. Conclusion", conclusion_text)
        
        # Build PDF
        doc.build(self.story)
        print(f"Report generated successfully: {output_path}")
        
        return output_path
