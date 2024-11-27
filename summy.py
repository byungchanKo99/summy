import os
from moviepy import VideoFileClip
from openai import OpenAI
from pydub import AudioSegment
import math
from pathlib import Path
import inquirer

client = OpenAI(api_key="")


def select_video_file():
    """대화형 CLI를 통해 비디오 파일 선택"""
    current_path = Path.cwd()

    while True:
        # 현재 디렉토리의 모든 항목 가져오기
        items = list(current_path.iterdir())
        # MP4 파일과 디렉토리만 필터링
        choices = [
            str(item.relative_to(current_path)) + ('/' if item.is_dir() else '')
            for item in items
            if item.is_dir() or item.suffix.lower() == '.mp4'
        ]

        # 상위 디렉토리로 이동 옵션 추가
        if current_path != Path.home():
            choices.insert(0, '../')

        questions = [
            inquirer.List('path',
                         message=f'현재 위치: {current_path}\n파일이나 디렉토리를 선택하세요',
                         choices=choices,
                         carousel=True)
        ]

        answer = inquirer.prompt(questions)
        if not answer:  # Ctrl+C로 종료한 경우
            return None

        selected = current_path / answer['path']

        if selected.is_file():
            return str(selected)
        else:  # 디렉토리인 경우
            # '../'를 선택한 경우
            if answer['path'] == '../':
                current_path = current_path.parent
            else:
                current_path = selected

def extract_audio(video_path, audio_path):
    """비디오에서 오디오 추출"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()

def split_audio(audio_path, max_size_bytes=25000000):  # 25MB보다 약간 작게 설정
    """오디오 파일을 청크로 분할"""
    audio = AudioSegment.from_mp3(audio_path)
    chunks = []
    
    # 1ms당 바이트 크기 계산 (근사치)
    bytes_per_ms = os.path.getsize(audio_path) / len(audio)
    
    # 청크의 길이(ms) 계산
    chunk_length_ms = int(max_size_bytes / bytes_per_ms)
    
    # 전체 길이를 chunk_length_ms로 나누어 청크 개수 계산
    total_chunks = math.ceil(len(audio) / chunk_length_ms)
    
    for i in range(total_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start_time:end_time]
        
        # 임시 파일로 저장
        chunk_path = f"temp_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        
        # 생성된 파일이 제한 크기를 넘지 않는지 확인
        if os.path.getsize(chunk_path) > max_size_bytes:
            os.remove(chunk_path)
            # 더 작은 청크로 재시도
            smaller_chunk_length_ms = int(chunk_length_ms * 0.8)  # 20% 감소
            chunk = audio[start_time:start_time + smaller_chunk_length_ms]
            chunk.export(chunk_path, format="mp3")
        print(f"청크 {i} 생성 완료")
        chunks.append(chunk_path)
    
    return chunks

def transcribe_audio(audio_path):
    """오디오를 텍스트로 변환"""
    print("오디오 파일을 청크로 분할")
    chunks = split_audio(audio_path)
    
    # 각 청크의 텍스트를 저장할 리스트
    transcripts = []
    
    try:
        # 각 청크를 개별적으로 처리
        for i, chunk_path in enumerate(chunks):
            print(f"청크 {i} 처리 중...")
            with open(chunk_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko"
                )
                transcripts.append(transcript.text)

        # 모든 텍스트를 하나로 합침
        print("모든 청크 처리 완료")
        return " ".join(transcripts)
    
    finally:
        # 임시 청크 파일들 삭제
        for chunk_path in chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

def split_text_into_chunks(text, max_tokens=4000):
    """텍스트를 토큰 크기에 맞게 분할"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    # 대략적으로 단어 1개를 1.3 토큰으로 계산
    words_per_chunk = int(max_tokens / 1.3)
    
    for word in words:
        current_chunk.append(word)
        current_length += 1
        
        if current_length >= words_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text):
    """텍스트 요약"""
    chunks = split_text_into_chunks(text)
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"청크 {i} 요약 중...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 전문적인 교육 멘토링 보고서 작성자입니다. 멘토링 세션의 녹음 내용을 명확하고 구조화된 보고서로 작성해주세요.

                    다음 형식으로 작성해주시되, 각 섹션을 상세하게 작성해주세요:

                    요약: [전체 멘토링 세션의 핵심 내용을 한 문장으로 간단명료하게 작성]

                    진행도: [
                    - 완료한 강의 번호
                    - 학습한 단원명
                    - 진행한 문제 유형이나 교재 정보
                    ]

                    멘토링 내용: [
                    - 학습한 주요 개념 설명
                    - 특별히 중점을 둔 학습 방법이나 전략
                    - 학생의 이해도 향상을 위해 사용한 구체적인 예시나 설명
                    - 발생한 어려움과 그 해결 방법
                    ]

                    시간별 진행 내용: [
                    구체적인 시간대별 활동을 다음과 같이 기록
                    XX분~XX분: [진행한 활동]
                    ]

                    예시:
                    요약: 도형의 닮음 단원 진도 나갔습니다.
                    진행도: 33강 ~ 36강, RPM 도형의 닮음 대표 유형 문제
                    멘토링 내용: 도형의 닮음은 도형의 합동에서 확장된 개념입니다. 그래서 도형의 합동을 바탕으로 닮음 문제를 풀었습니다. 이번 시간엔 문제를 풀며 특히 개념이 헷갈릴 때 어떻게 대처할 수 있는지에 대해 다뤘습니다...
                    시간별 진행 내용:
                    00분~05분: 근황 토크
                    05분~55분: 문제풀이
                    55분~60분: 다음 숙제 안내"""
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ],
            max_tokens=1000
        )
        chunk_summaries.append(response.choices[0].message.content)
    
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    final_text = " ".join(chunk_summaries)
    print("청크 요약 완료")
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """당신은 전문적인 교육 멘토링 보고서 작성자입니다. 멘토링 세션의 녹음 내용을 명확하고 구조화된 보고서로 작성해주세요.

                다음 형식으로 작성해주시되, 각 섹션을 상세하게 작성해주세요:

                요약: [전체 멘토링 세션의 핵심 내용을 한 문장으로 간단명료하게 작성]

                진행도: [
                - 완료한 강의 번호
                - 학습한 단원명
                - 진행한 문제 유형이나 교재 정보
                ]

                멘토링 내용: [
                - 학습한 주요 개념 설명
                - 특별히 중점을 둔 학습 방법이나 전략
                - 학생의 이해도 향상을 위해 사용한 구체적인 예시나 설명
                - 발생한 어려움과 그 해결 방법
                ]

                시간별 진행 내용: [
                구체적인 시간대별 활동을 다음과 같이 기록
                XX분~XX분: [진행한 활동]
                ]

                예시:
                요약: 도형의 닮음 단원 진도 나갔습니다.
                진행도: 33강 ~ 36강, RPM 도형의 닮음 대표 유형 문제
                멘토링 내용: 도형의 닮음은 도형의 합동에서 확장된 개념입니다. 그래서 도형의 합동을 바탕으로 닮음 문제를 풀었습니다. 이번 시간엔 문제를 풀며 특히 개념이 헷갈릴 때 어떻게 대처할 수 있는지에 대해 다뤘습니다...
                시간별 진행 내용:
                00분~05분: 근황 토크
                05분~55분: 문제풀이
                55분~60분: 다음 숙제 안내"""
            },
            {
                "role": "user",
                "content": final_text
            }
        ],
        max_tokens=1000
    )
    print("최종 요약 완료")
    return final_response.choices[0].message.content

def main():
    # CLI로 비디오 파일 선택
    video_path = select_video_file()
    if not video_path:
        print("파일이 선택되지 않았습니다.")
        return

    audio_path = "temp_audio.mp3"

    try:
        # 1. 비디오에서 오디오 추출
        print(f"선택된 파일: {video_path}")
        print("오디오 추출 중...")
        extract_audio(video_path, audio_path)

        # 2. 오디오를 텍스트로 변환
        print("음성을 텍스트로 변환 중...")
        transcript = transcribe_audio(audio_path)

        # 3. 텍스트 요약
        print("텍스트 요약 중...")
        summary = summarize_text(transcript)

        # 4. 결과 출력
        print("\n=== 요약 결과 ===")
        print(summary)

    finally:
        # 임시 오디오 파일 삭제
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    main()
