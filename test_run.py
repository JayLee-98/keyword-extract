# test_run.py
import asyncio
from test import request_ai_result, request_at_keyword_list, request_at_embbeding

async def main():
    # 테스트용 텍스트
    text = """
    인공지능(AI)은 인간의 지능을 모방하여 학습, 추론, 지각, 언어이해 등을 컴퓨터로 구현하는 기술이다. 
    머신러닝은 AI의 한 분야로, 데이터로부터 패턴을 학습하여 새로운 데이터에 대해 예측이나 결정을 내린다. 
    딥러닝은 머신러닝의 한 방법으로, 인간의 신경망을 모방한 인공신경망을 사용하여 복잡한 패턴을 학습한다.
    최근에는 자연어처리 분야에서 큰 발전이 있었으며, BERT나 GPT와 같은 모델들이 좋은 성능을 보이고 있다.
    """
    
    print("=== 전체 분석 시작 ===")
    result = await request_ai_result(text)
    
    print(f"\n[전체 문서 제목]\n{result.result.title}")
    
    print("\n[청크별 분석 결과]")
    for i, chunk in enumerate(result.result.chunc_list, 1):
        print(f"\n청크 {i}:")
        print(f"제목: {chunk.title}")
        print(f"요약: {chunk.summary}")
        print(f"키워드: {', '.join(chunk.keyword_list)}")
        print(f"임베딩 차원: {len(chunk.embbeding)}")
    
    print("\n=== 키워드 추출만 테스트 ===")
    keywords = await request_at_keyword_list(text)
    print(f"추출된 키워드: {', '.join(keywords)}")
    
    print("\n=== 임베딩 추출 테스트 ===")
    embedding = await request_at_embbeding(text)
    print(f"임베딩 차원: {len(embedding)}")

if __name__ == "__main__":
    asyncio.run(main())