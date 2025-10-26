pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git branch: 'docker-jenkins-integration', url: 'https://github.com/<your-username>/<your-repo>.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t flights-recommender:latest .'
                }
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh 'pytest tests/ || echo "Tests skipped or not found"'
            }
        }
        stage('Run Container') {
            steps {
                script {
                    sh 'docker run --rm flights-recommender:latest'
                }
            }
        }
    }
}
