# Literature Survey — References

## 1. RAG Foundations

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. arXiv:2005.11401
   - Foundational RAG paper — combines a pre-trained seq2seq model with dense retrieval over external knowledge. Core architecture this dissertation builds on.

2. Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). "REALM: Retrieval-Augmented Language Model Pre-Training." *ICML 2020*. arXiv:2002.08909
   - Precursor to RAG — integrates a latent knowledge retriever into LM pre-training. Establishes theoretical basis for retrieval-augmented approaches.

3. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., and Wang, H. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv preprint*. arXiv:2312.10997
   - Comprehensive RAG survey covering Naive RAG, Advanced RAG, Modular RAG paradigms. Useful for positioning our architecture.

## 2. Domain-Adapted LLMs

4. Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D., and Mann, G. (2023). "BloombergGPT: A Large Language Model for Finance." *arXiv preprint*. arXiv:2303.17564
   - 50B-parameter model trained on 363B tokens of financial data. Demonstrates domain-specific pre-training yields superior domain performance without sacrificing general capability.

5. Singhal, K., Azizi, S., Tu, T., et al. (2023). "Large Language Models Encode Clinical Knowledge." *Nature*, 620(7972), 172-180. DOI:10.1038/s41586-023-06291-2
   - Med-PaLM — first LLM to exceed passing scores on USMLE medical questions. Demonstrates domain adaptation via instruction tuning for a specialized professional domain.

6. Colombo, P., Pires, T.P., Boudiaf, M., et al. (2024). "SaulLM-7B: A Pioneering Large Language Model for Law." *arXiv preprint*. arXiv:2403.03883
   - First LLM designed for legal text, built on Mistral 7B with 30B+ legal tokens. Precedent for adapting a small open-source model to a professional domain.

## 3. Small/Efficient LLMs

7. Qwen Team (Yang, A., Yang, B., Hui, B., Zheng, B., et al.) (2024). "Qwen2.5 Technical Report." *arXiv preprint*. arXiv:2412.15115
   - Technical report for Qwen2.5 family (0.5B-72B). Architecture details and benchmarks for the 3B and 7B models used in this dissertation.

8. Xu, J., Li, Z., Chen, W., Wang, Q., Gao, X., Cai, Q., and Ling, Z. (2024). "On-Device Language Models: A Comprehensive Review." *arXiv preprint*. arXiv:2409.00088
   - Surveys efficient architectures, compression, and deployment for resource-constrained devices. Relevant to small-model hospitality deployment argument.

9. Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. arXiv:2210.17323
   - One-shot weight quantization enabling large models on single GPUs with minimal accuracy loss. Foundational for Q4_K_M quantized variants used here.

## 4. LLM Evaluation

10. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023 (Datasets and Benchmarks Track)*. arXiv:2306.05685
    - Establishes LLM-as-judge evaluation paradigm. Directly relevant to our Grok-based scoring methodology.

11. Akkiraju, R., et al. (2024). "FACTS About Building Retrieval Augmented Generation-based Chatbots." *arXiv preprint (NVIDIA)*. arXiv:2407.07858
    - Enterprise RAG chatbot framework (Freshness, Architectures, Cost, Testing, Security) with empirical accuracy-latency tradeoffs. Applicable to our enterprise RAG system design.

## 5. Hospitality / Hotel Technology

12. Pan, T. and Fu, R.J. (2026). "Navigating the AI Horizon in Hospitality: A Novel Classification and Future Research Agenda." *International Hospitality Review*, 40(1), 81-110. DOI:10.1108/IHR-01-2024-0003
    - Classification of AI applications in hospitality (prediction, CV, NLP, behavioral research). Positions our work within the broader hospitality AI landscape.

## Gaps / To Find

- Hospitality NLP/chatbot papers: search "chatbot hotel guest service NLP" in Google Scholar
- PMS-specific papers: search "property management system hotel cloud" in Google Scholar
- ENTER (eTourism) conference proceedings, International Journal of Hospitality Management
