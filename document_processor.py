import os
import re
import json
from datetime import datetime
from docling.document_converter import DocumentConverter
from typing import Dict, List


class DocumentProcessor:
    def __init__(self):
        self.converter = DocumentConverter()
    
    def pdf_names(self, directory_path: str) -> List[str]:
        files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        return files
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'-{2,}', '--', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def split_by_sections(self, text: str) -> List[Dict]:
        sections = []
        lines = text.split('\n')
        current_section = ""
        current_title = "Introduction"
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                # Önceki bölümü kaydet
                if current_section.strip():
                    sections.append({
                        'title': current_title,
                        'content': current_section.strip(),
                        'header_level': len(header_match.group(1))
                    })
                
                current_title = header_match.group(2).strip()
                current_section = ""
            else:
                current_section += line + '\n'
        
        # Son bölümü kaydet
        if current_section.strip():
            sections.append({
                'title': current_title,
                'content': current_section.strip(),
                'header_level': 1
            })
        
        return sections
    
    def read_pdf(self, pdf_path: str) -> List[Dict]:
        """PDF'yi okuyup LLM-ready formata çevir"""
        try:
            # PDF'yi Markdown'a çevir
            result = self.converter.convert(pdf_path)
            markdown_text = result.document.export_to_markdown()
            
            # Metni temizle
            clean_text = self.clean_text(markdown_text)
            
            # Bölümlere ayır
            sections = self.split_by_sections(clean_text)
            
            # LLM için optimize edilmiş format
            processed_sections = []
            for i, section in enumerate(sections, 1):
                processed_sections.append({
                    'section_id': i,
                    'title': section['title'],
                    'content': section['content'],
                    'header_level': section.get('header_level', 1),
                    'word_count': len(section['content'].split()),
                    'char_count': len(section['content']),
                    # LLM context için tam metin
                    'full_context': f"# {section['title']}\n\n{section['content']}"
                })
            
            return processed_sections
            
        except Exception as e:
            print(f"Hata: {pdf_path} işlenirken - {str(e)}")
            return []
    
    def save_to_json(self, data: List[Dict], filename: str = None) -> str:
        """LLM-ready JSON formatında kaydet"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_ready_{timestamp}.json"
        
        output_data = {
            'processed_at': datetime.now().isoformat(),
            'total_sections': len(data),
            'format': 'LLM-optimized',
            'sections': data,
            # LLM için tüm içeriği birleştirilmiş hali
            'full_document': '\n\n'.join([section['full_context'] for section in data])
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ LLM-ready veri kaydedildi: {filename}")
        return filename


def process_and_save():
    """Ana işlem - PDF'leri LLM-ready formata çevir"""
    processor = DocumentProcessor()
    
    directory = "data"  
    files = processor.pdf_names(directory)
    
    if not files:
        print("❌ PDF dosyası bulunamadı!")
        return
    
    print(f"📄 Bulunan PDF'ler: {files}")
    
    for pdf_file in files:
        print(f"\n🔄 İşleniyor: {pdf_file}")
        
        sections = processor.read_pdf(os.path.join(directory, pdf_file))
        
        if sections:
            clean_name = pdf_file.replace('.pdf', '')
            filename = f"llm_ready_{clean_name}.json"
            
            processor.save_to_json(sections, filename)
            print(f"✅ {len(sections)} bölüm başarıyla işlendi")
        else:
            print(f"❌ {pdf_file} işlenemedi")


if __name__ == "__main__":
    process_and_save()

