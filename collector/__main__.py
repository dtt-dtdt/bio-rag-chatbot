"""python -m collector  →  원천 데이터 확인 (store/raw/)"""
from collector import storage

if __name__ == "__main__":
    print("[원천 데이터 확인] store/raw/")
    if storage.check():
        print("\n다음 단계: python -m indexer chunk")
    else:
        print("\nREADME의 '데이터 준비'를 보고 파일을 store/raw/에 넣어 주세요.")
